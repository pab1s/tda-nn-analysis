from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FakeData, ImageFolder
from datasets.car_dataset import CarDataset

def get_dataset(name, root_dir, train=None, transform=None):
    """
    Returns a dataset based on the given name.

    Args:
        name (str): The name of the dataset.
        root_dir (str): The root directory where the dataset is stored.
        train (bool, optional): If True, returns the training set. If False, returns the test set. Defaults to None.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.

    Returns:
        torch.utils.data.Dataset: The requested dataset.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    if name == 'CIFAR10':
        return CIFAR10(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'CIFAR100':
        return CIFAR100(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'MNIST':
        return MNIST(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'Imagenette':
        return ImageFolder(root=root_dir, transform=transform)
    elif name == 'FakeData':
        return FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transform)
    elif name == 'CarDataset':
        return CarDataset(root_dir, train=train, transform=transform)
    else:
        raise ValueError(f"Dataset {name} not supported.")
