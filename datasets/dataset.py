from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FakeData
from datasets.car_dataset import CarDataset

def get_dataset(name, root_dir, train=None, transform=None):
    if name == 'CIFAR10':
        return CIFAR10(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'CIFAR100':
        return CIFAR100(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'MNIST':
        return MNIST(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'FakeData':
        return FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transform)
    elif name == 'CarDataset':
        return CarDataset(root_dir, train=train, transform=transform)
    else:
        raise ValueError(f"Dataset {name} not supported.")
