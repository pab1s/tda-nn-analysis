import pytest
from torchvision import transforms, datasets
from datasets.dataset import get_dataset

@pytest.fixture
def basic_transform():
    return transforms.Compose([transforms.ToTensor()])

def test_cifar10_download_and_load(basic_transform):
    """
    Test function to download and load CIFAR10 dataset.

    Args:
        basic_transform: A transformation to apply to the dataset.

    Returns:
        None

    Raises:
        AssertionError: If the train dataset is not an instance of datasets.CIFAR10.
        AssertionError: If the test dataset is not an instance of datasets.CIFAR10.
        AssertionError: If the CIFAR10 train dataset does not contain 50,000 images.
        AssertionError: If the CIFAR10 test dataset does not contain 10,000 images.
    """
    root_dir = './data'
    train_dataset = get_dataset('CIFAR10', root_dir=root_dir, train=True, transform=basic_transform)
    test_dataset = get_dataset('CIFAR10', root_dir=root_dir, train=False, transform=basic_transform)

    assert isinstance(train_dataset, datasets.CIFAR10), "The train dataset should be an instance of datasets.CIFAR10"
    assert isinstance(test_dataset, datasets.CIFAR10), "The test dataset should be an instance of datasets.CIFAR10"

    # CIFAR10 dataset specific checks
    assert len(train_dataset) == 50000, "The CIFAR10 train dataset should contain 50,000 images."
    assert len(test_dataset) == 10000, "The CIFAR10 test dataset should contain 10,000 images."

def test_unsupported_dataset(basic_transform):
    root_dir = './data'
    with pytest.raises(ValueError):
        _ = get_dataset('UnsupportedDataset', root_dir=root_dir, train=True, transform=basic_transform)
