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
        self.activations = None
        self.gradients = None
        self.linear.weight.register_hook(lambda grad: setattr(self, 'gradients', grad) or grad)

    def forward(self, x):
        out = self.linear(x)
        self.activations = out.detach()
        return out

    @torch.no_grad()
    def update_influence(self):
        if self.activations is not None and self.gradients is not None:
            inf = (self.activations.abs().mean(0) * self.gradients.abs().mean(1))
            self.influence_score = self.decay * self.influence_score + (1 - self.decay) * inf

    @torch.no_grad()
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

def get_split_mnist(digits):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    indices = [i for i, (_, label) in enumerate(dataset) if label in digits]
    return DataLoader(Subset(dataset, indices), batch_size=128, shuffle=True)

def evaluate_task(model, loader, device, allowed_digits):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            mask = torch.full_like(output, float('-inf'))
            mask[:, allowed_digits] = 0
            masked_output = output + mask
            pred = masked_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    return correct / total

# --- Main Scan ---

def run_fine_scan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strengths = np.linspace(0.001, 0.15, 15).tolist()
    strengths.append(1.0)
    
    results = []
    hidden_size = 64
    loader_a = get_split_mnist([0, 1, 2, 3, 4])
    loader_b = get_split_mnist([5, 6, 7, 8, 9])

    print(f"START FINE-SCAN (Hidden Size: {hidden_size})")

    for iso in strengths:
        print(f"Scanning Iso: {iso:.4f}...", end=" ", flush=True)
        model = LiquidNet(28*28, hidden_size, 10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        for _ in range(2):
            model.train()
            for batch_idx, (data, target) in enumerate(loader_a):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad(); loss = criterion(model(data), target); loss.backward()
                model.apply_gating()
                model.classifier.weight.grad[5:] = 0; model.classifier.bias.grad[5:] = 0
                optimizer.step()
                model.layer1.update_influence(); model.layer2.update_influence()
                if batch_idx % 5 == 0: model.layer1.evolve_structure(iso); model.layer2.evolve_structure(iso)

        for _ in range(2):
            model.train()
            for batch_idx, (data, target) in enumerate(loader_b):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad(); loss = criterion(model(data), target); loss.backward()
                model.apply_gating()
                model.classifier.weight.grad[:5] = 0; model.classifier.bias.grad[:5] = 0
                optimizer.step()
                model.layer1.update_influence(); model.layer2.update_influence()
                if batch_idx % 5 == 0: model.layer1.evolve_structure(iso); model.layer2.evolve_structure(iso)

        acc_a = evaluate_task(model, loader_a, device, [0,1,2,3,4])
        print(f"Acc A: {acc_a:.4f}")
        results.append((iso, acc_a))

    # Plot
    iso_vals, a_vals = zip(*results)
    plt.figure(figsize=(12, 6))
    plt.plot(iso_vals[:-1], a_vals[:-1], 'g-o', label='Task A Resilienz')
    plt.axhline(y=a_vals[-1], color='r', linestyle='--', label='Baseline (Iso=1.0)')
    plt.title(f'Fine-Analysis of the Isolation Strength (Hidden Size: {hidden_size})')
    plt.xlabel('Isolation Strength')
    plt.ylabel('Memory Accuracy Task A')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_fine_scan()
