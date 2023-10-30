import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from simple_cnn import SimpleCNN
import os
import torch
import torch.nn.functional as F
from simple_cnn import SimpleCNN
import neptune
from dotenv import load_dotenv

# Load environment variables from .env
# load_dotenv()

run = neptune.init_run(
    project="krishna.rastogi/imageclassificationcifar10",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMzViNzZjYi1lYzg0LTQwMjMtOWRhNi02ZDljYWQ1MjJmMDEifQ==",
)  # your credentials

# Load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = datasets.CIFAR10(root='../../data/raw', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.CIFAR10(root='../../data/raw', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics to Neptune
        if batch_idx % 100 == 0:
            neptune.log_metric('training_loss', loss.item())

    # Validation loop and metrics logging
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    
    neptune.log_metric('validation_loss', val_loss)
    neptune.log_metric('validation_accuracy', val_accuracy)

# Save the trained model
torch.save(model.state_dict(), '../../models/cifar10_simple_cnn.pth')
neptune.log_artifact('model.pt', model.state_dict())
neptune.stop()
