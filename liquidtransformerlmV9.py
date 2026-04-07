import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math

# --- 1. Liquid V9: Synaptic Pressure Layer ---

class LiquidLinearFuzzy(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.18):
        self.line = nn.Linear(in_features, out_features, bias=False)
        self.sigma = sigma 
        self.register_buffer('influence_score', torch.zeros(out_features))
        self.register_buffer('soft_mask', torch.ones(out_features))
        self.activations = None
        self.gradients = None
        self.line.weight.register_hook(lambda grad: setattr(self, 'gradients', grad) or grad)

    def update_metrics(self):
        if self.activations is not None and self.gradients is not None:
            act_imp = self.activations.abs().mean(dim=list(range(self.activations.dim()-1)))
            grad_imp = self.gradients.abs().mean(dim=1)
            self.influence_score = 0.90 * self.influence_score + 0.10 * (act_imp * grad_imp)

    @torch.no_grad()
    def evolve_gaussian(self):
        s = self.influence_score
        ranks = torch.argsort(torch.argsort(s)).float() / (len(s) - 1)
        new_mask = torch.exp(-((1.0 - ranks)**2) / (2 * self.sigma**2))
        self.soft_mask = new_mask 

    def forward(self, x):
        self.activations = self.line(x)
        return self.activations

# --- 2. Architecture (V9: Triple Head) ---

class LiquidBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.qkv = LiquidLinearFuzzy(d_model, 3 * d_model)
        self.attn_out = LiquidLinearFuzzy(d_model, d_model)
        self.ffn = LiquidLinearFuzzy(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)

    def forward(self, x, mask=None):
        nx = self.ln1(x)
        B, T, C = nx.size()
        qkv_out = self.qkv(nx) 
        qkv = qkv_out.reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if mask is not None: att = att.masked_fill(mask == float('-inf'), float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.attn_out(y)
        x = x + self.ffn_out(F.gelu(self.ffn(self.ln2(x))))
        return x

class TripleHeadLiquidLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 512, d_model))
        self.layers = nn.ModuleList([LiquidBlock(d_model, nhead, d_model*4) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        
        self.head_story = LiquidLinearFuzzy(d_model, vocab_size)
        self.head_code  = LiquidLinearFuzzy(d_model, vocab_size)
        self.head_wiki  = LiquidLinearFuzzy(d_model, vocab_size)

    def forward(self, idx, head_idx=0):
        x = self.token_emb(idx) + self.pos_emb[:, :idx.size(1), :]
        mask = torch.triu(torch.ones(idx.size(1), idx.size(1), device=idx.device) * float('-inf'), 1)
        for layer in self.layers: x = layer(x, mask=mask)
        x = self.ln_f(x)
        if head_idx == 0: return self.head_story(x)
        if head_idx == 1: return self.head_code(x)
        return self.head_wiki(x)

# --- 3. Master Training Logic ---

def train_step(model, batch, optimizer, scaler, phase_idx):
    x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    optimizer.zero_grad()
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x, head_idx=phase_idx)
        base_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        penalty = 0

        if phase_idx > 0:
            for name, m in model.named_modules():
                if isinstance(m, LiquidLinearFuzzy) and "layers" in name:
                    
                    penalty += 0.15 * (m.activations.abs() * torch.pow(m.soft_mask, 4.0)).mean()
        
        total_loss = base_loss + penalty

    scaler.scale(total_loss).backward()
    

    if phase_idx > 0:
        for name, m in model.named_modules():
            if isinstance(m, LiquidLinearFuzzy):
                if "head_story" in name: 
                    if m.line.weight.grad is not None: m.line.weight.grad.zero_()
                elif "layers" in name:
                    grad_gate = 1.0 - torch.pow(m.soft_mask, 4.0)
                    if m.line.weight.grad is not None: m.line.weight.grad *= grad_gate.unsqueeze(1)

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LiquidLinearFuzzy):
                m.update_metrics()
                if phase_idx == 0: m.evolve_gaussian()
                
    return base_loss.item()

# --- 4. DIAGNOSIS & Multi-Task Generation ---

def analyze_model(model):
    print("\n" + "="*50)
    print("DIAGNOSIS: MULTI-TASK NEURAL DISTRIBUTION")
    print("="*50)
    for name, m in model.named_modules():
        if isinstance(m, LiquidLinearFuzzy) and "layers" in name:
            s = m.influence_score
            ranks = torch.argsort(torch.argsort(s)).float() / (len(s) - 1)
            active_usage = (ranks > 0.8).float()
            core_overlap = (active_usage * (m.soft_mask > 0.85).float()).sum()
            transfer_zone = (active_usage * ((m.soft_mask <= 0.85) & (m.soft_mask > 0.15)).float()).sum()
            print(f"{name:20} | Core-Protection: {core_overlap.item():3.0f} | Transfer-Zone: {transfer_zone.item():3.0f}")

def generate_multi(model, tokenizer, prompt, device, head=0):
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for _ in range(60):
        with torch.no_grad():
            logits = model(ids, head_idx=head)[:, -1, :] / 0.75
            v, _ = torch.topk(logits, 40)
            logits[logits < v[:, [-1]]] = -float('Inf')
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id: break
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# --- 5. Main Execution ---

def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = TripleHeadLiquidLM(len(tokenizer)).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=1e-4) 
    scaler = torch.amp.GradScaler('cuda')

    STEPS = 5000 
    
    stories = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    codes   = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
    try:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    except:
        wiki = load_dataset("HuggingFaceFW/clean-wikipedia", split="train", streaming=True)
    
    def get_loader(ds, task_type='story'):
        def collate(batch):
            if task_type == 'story': 
                texts = [b['text'] for b in batch]
            elif task_type == 'code': 
                texts = [f"# {b['instruction']}\n{b['output']}" for b in batch]
            else: 
                texts = [b['text'][:1024] for b in batch] 
            
            return tokenizer(texts, truncation=True, max_length=128, padding="max_length", return_tensors="pt")["input_ids"]
        
        return DataLoader(ds, batch_size=16, collate_fn=collate)

    loaders = [get_loader(stories, 'story'), get_loader(codes, 'code'), get_loader(wiki, 'wiki')]
    names = ["STORIES (Head 0)", "PYTHON (Head 1)", "WIKIPEDIA (Head 2)"]

    for phase in range(3):
        it = iter(loaders[phase])
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].zero_()

        for i in tqdm(range(STEPS), desc=f"Training {names[phase]}"):
            try: batch = next(it)
            except StopIteration: it = iter(loaders[phase]); batch = next(it)
            train_step(model, batch.to(device), optimizer, scaler, phase_idx=phase)

    analyze_model(model)
    
    print("\n--- LIQUID TEST (V9 SGD-Edition) ---")
    print(f"HEAD 0 (Story): {generate_multi(model, tokenizer, 'Once upon a time, a small bird', device, head=0)}")
    print("-" * 50)
    print(f"HEAD 1 (Code) : {generate_multi(model, tokenizer, 'def find_max(list):', device, head=1)}")
    print("-" * 50)
    print(f"HEAD 2 (Wiki) : {generate_multi(model, tokenizer, 'The capital city is', device, head=2)}")

if __name__ == "__main__":
    main()