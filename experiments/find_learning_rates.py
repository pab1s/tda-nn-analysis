import torch
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from utils.training import find_lr
from utils.plotting import plot_lr_vs_loss
from factories.model_factory import ModelFactory
from factories.loss_factory import LossFactory
from factories.optimizer_factory import OptimizerFactory
from os import path

def main(config_path, optimizer_type, optimizer_params, batch_size):
    """
    Main function for finding learning rates.

    Args:
        config_path (str): The path to the configuration file.
        optimizer_type (str): The type of optimizer to use.
        optimizer_params (dict): The parameters for the optimizer.
        batch_size (int): The batch size for the data loader.

    Returns:
        None
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and transform data
    transforms = get_transforms(config['data']['transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    # Split data
    total_size = len(data)
    test_size = int(total_size * config['data']['test_size'])
    val_size = int(total_size * config['data']['val_size'])
    train_size = total_size - test_size - val_size

    data_train, _ = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, _ = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    # Data loaders using the given batch_size
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    # Model setup
    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], **config['model']['parameters']).to(device)

    # Loss setup
    loss_factory = LossFactory()
    criterion = loss_factory.create(config['training']['loss_function']['type'])

    # Optimizer setup with given parameters
    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create(optimizer_type, params=model.parameters(), **optimizer_params)

    # Find learning rate
    print("Finding learning rate...")
    log_lrs, losses = find_lr(model, train_loader, criterion, optimizer, optimizer_params, device=device)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_filename = f"lr_vs_loss_{config['model']['type']}_{current_time}_batch{batch_size}_{optimizer_type}.png"
    plot_lr_vs_loss(log_lrs, losses, path.join(config['paths']['plot_path'], plot_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process configuration file, optimizer types, and batch sizes.')
    parser.add_argument('config_filename', type=str, help='Filename of the configuration file within the "config" directory')

    args = parser.parse_args()

    batch_sizes = [64]
    optimizer_types = ["SGD"]
    adam_params = {
        "lr": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False
    }
    sgd_params = {
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0,
        "nesterov": False
    }

    config_path = f"config/{args.config_filename}"

    for optimizer_type in optimizer_types:
        for batch_size in batch_sizes:
            optimizer_params = adam_params if optimizer_type == "Adam" else sgd_params
            main(config_path, optimizer_type, optimizer_params, batch_size)
