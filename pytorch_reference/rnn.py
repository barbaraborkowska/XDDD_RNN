import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time
import psutil

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

def loader(dataset, batch_size=1):
    x = dataset.data.view(-1, 28 * 28).float() / 255.0
    y = dataset.targets
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = loader(train_dataset, batch_size=100)
test_loader = loader(test_dataset, batch_size=len(test_dataset))

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(14 * 14, 64, batch_first=True)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 4, 14 * 14)
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RNNModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=15e-3)

def loss_and_accuracy(model, loader):
    model.eval()
    if device == 'cuda':
        torch.cuda.reset_max_memory_allocated(device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).float().mean().item()
    memory_usage = torch.cuda.max_memory_allocated(device) if device == 'cuda' else psutil.Process().memory_info().rss
    return loss.item(), accuracy * 100, memory_usage

start_time = time.time()
if device == 'cuda':
    torch.cuda.reset_max_memory_allocated(device)
train_memory_start = psutil.Process().memory_info().rss if device == 'cpu' else None

num_epochs = 5
train_log = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

train_time = time.time() - start_time
train_memory_end = psutil.Process().memory_info().rss if device == 'cpu' else torch.cuda.max_memory_allocated(device)

train_loss, train_acc, _ = loss_and_accuracy(model, train_loader)
test_loss, test_acc, test_memory = loss_and_accuracy(model, test_loader)

print(f'Total Training Time: {train_time:.2f}s')
print(f'Total Training Memory: {(train_memory_end - train_memory_start)/1024**2:.2f}MB' if device == 'cpu' else f'Total Training Memory: {train_memory_end/1024**2:.2f}MB')
print(f'Test Memory: {test_memory/1024**2:.2f}MB')
print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
