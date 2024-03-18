from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform

def get_dataset(name, root_dir, train, transform):
    if name == 'CIFAR10':
        return datasets.CIFAR10(root=root_dir, train=train, download=True, transform=transform)
    elif name == 'CustomDataset':
        return CustomDataset(root_dir, train=train, transform=transform)
    else:
        raise ValueError(f"Dataset {name} not supported.")

def get_dataloaders(config):
    transform = get_transform()
    train_dataset = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transform)
    test_dataset = get_dataset(config['data']['name'], config['data']['dataset_path'], train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, test_loader
