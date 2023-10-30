import torchvision
from torchvision import datasets, transforms

def download_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Download CIFAR-10 datasets
    datasets.CIFAR10(root='../../data/raw', train=True, download=True, transform=transform)
    datasets.CIFAR10(root='../../data/raw', train=False, download=True, transform=transform)

if __name__ == "__main__":
    download_cifar10()
