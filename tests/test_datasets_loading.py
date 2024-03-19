import pytest
from utils.data_utils import get_dataloaders

def test_load_cifar10():
    config = {
        'data': {
            'name': 'CIFAR10',
            'dataset_path': './data',
        },
        'training': {
            'batch_size': 4,
        }
    }
    train_loader, test_loader = get_dataloaders(config)
    # Check that loaders are not empty
    assert len(train_loader) > 0, "CIFAR10 training loader should not be empty"
    assert len(test_loader) > 0, "CIFAR10 test loader should not be empty"

