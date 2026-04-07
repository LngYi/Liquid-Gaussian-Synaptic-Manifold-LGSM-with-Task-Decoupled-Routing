import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- Architecture ---
class LiquidLayer(nn.Module):
    def __init__(self, in_features, out_features, decay=0.9):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.decay = decay
        self.register_buffer('influence_score', torch.zeros(out_features))
        self.register_buffer('soft_mask', torch.ones(out_features))
        self.activations, self.gradients = None, None
        self.linear.weight.register_hook(lambda grad: setattr(self, 'gradients', grad) or grad)

    def forward(self, x):
        out = self.linear(x)
        self.activations = out.detach()
        return out

    def update_influence(self):
        if self.activations is not None and self.gradients is not None:
            inf = (self.activations.abs().mean(0) * self.gradients.abs().mean(1))
            self.influence_score = self.decay * self.influence_score + (1 - self.decay) * inf

    def evolve_structure(self, iso, top_k_percent=0.25):
        k = max(1, int(len(self.influence_score) * top_k_percent))
        _, top_indices = torch.topk(self.influence_score, k)
        new_mask = torch.full_like(self.soft_mask, iso)
        new_mask[top_indices] = 1.0
        self.soft_mask = new_mask

class LiquidNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.layer1 = LiquidLayer(input_size, hidden_size)
        self.layer2 = LiquidLayer(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.classifier(x)

    def apply_gating(self):
        for l in [self.layer1, self.layer2]:
            if l.linear.weight.grad is not None:
                l.linear.weight.grad *= l.soft_mask.unsqueeze(1)
                l.linear.bias.grad *= l.soft_mask

# --- Functions ---
def get_loaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    def split(digits):
        indices = [i for i, (_, label) in enumerate(train_set) if label in digits]
        return DataLoader(Subset(train_set, indices), batch_size=128, shuffle=True)
    
    return split([0,1,2,3,4]), split([5,6,7,8,9])

def evaluate_task(model, loader, device, allowed_digits):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            mask = torch.full_like(output, float('-inf'))
            mask[:, allowed_digits] = 0
            pred = (output + mask).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    return correct / total

# --- Experiment Loop with Multi-Seed ---
def run_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [0.001, 0.033, 0.065, 0.107, 1.0]
    seeds = [42, 1337, 7]
    
    hidden_size = 64
    loader_a, loader_b = get_loaders()
    
    final_results = {}

    print(f"START VALIDATION (Hidden Size: {hidden_size}, Seeds: {len(seeds)})")

    for iso in candidates:
        seed_accs = []
        print(f"\nTest ISO {iso:.4f}:", end=" ", flush=True)
        
        for seed in seeds:
            torch.manual_seed(seed)
            model = LiquidNet(28*28, hidden_size, 10).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.1)
            criterion = nn.CrossEntropyLoss()

            # Task A & B
            for loader, task_range in [(loader_a, range(0,5)), (loader_b, range(5,10))]:
                for _ in range(2):
                    model.train()
                    for batch_idx, (data, target) in enumerate(loader):
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        loss = criterion(model(data), target)
                        loss.backward()
                        model.apply_gating()
                        
                        # Classifier Gating
                        if task_range == range(0,5):
                            model.classifier.weight.grad[5:] = 0; model.classifier.bias.grad[5:] = 0
                        else:
                            model.classifier.weight.grad[:5] = 0; model.classifier.bias.grad[:5] = 0
                            
                        optimizer.step()
                        model.layer1.update_influence(); model.layer2.update_influence()
                        if batch_idx % 5 == 0:
                            model.layer1.evolve_structure(iso); model.layer2.evolve_structure(iso)

            acc_a = evaluate_task(model, loader_a, device, [0,1,2,3,4])
            seed_accs.append(acc_a)
            print(f"{acc_a:.2f}", end=" ", flush=True)
        
        final_results[iso] = seed_accs

    # --- Analyse & Plot ---
    labels = [str(c) for c in candidates]
    means = [np.mean(final_results[c]) for c in candidates]
    stds = [np.std(final_results[c]) for c in candidates]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=stds, capsize=10, color=['gray', 'green', 'green', 'green', 'red'])
    plt.title(f'Validation of the isolation (average over {len(seeds)} Seeds)')
    plt.ylabel('Memory Accuracy Task A')
    plt.xlabel('Isolation Strength')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    best_iso = candidates[np.argmax(means)]
    print(f"\nRESULT: The most stable optimum is at ISO = {best_iso}")

if __name__ == "__main__":
    run_validation()