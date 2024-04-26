import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from utils.metrics import Accuracy, Precision
from factories.model_factory import ModelFactory
from factories.loss_factory import LossFactory
from factories.optimizer_factory import OptimizerFactory
from trainers import get_trainer
from os import path

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and transform data
    transforms = get_transforms(config['data']['transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    # Split data
    total_size = len(data)
    test_size = int(total_size * config['data']['test_size'])
    val_size = int((total_size - test_size) * config['data']['val_size'])
    train_size = total_size - test_size - val_size

    data_train, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, data_val = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    # Data loaders
    train_loader = DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(data_val, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(data_test, batch_size=config['training']['batch_size'], shuffle=False)

    # Model setup
    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], **config['model']['parameters']).to(device)

    # Loss and optimizer setup
    loss_factory = LossFactory()
    criterion = loss_factory.create(config['training']['loss_function']['type'])

    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create(config['training']['optimizer']['type'], params=model.parameters(), **config['training']['optimizer']['parameters'])

    # Metrics and trainer setup
    metrics = [Accuracy(), Precision()]
    trainer = get_trainer(config['trainer'], model=model, device=device)

    # Training stages setup
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dataset_time = f"{config['model']['type']}_{config['data']['name']}_{current_time}"
    log_filename = path.join(config['paths']['log_path'], f"log_finetuning_{model_dataset_time}.csv")
    plot_filename = path.join(config['paths']['plot_path'], f"plot_finetuning_{model_dataset_time}.png")

    # Initial training stage
    print("Starting initial training stage with frozen layers...")
    trainer.build(
        criterion=criterion,
        optimizer=optimizer,
        freeze_until_layer=config['training']['freeze_until_layer'],
        metrics=metrics
    )
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['training']['epochs']['initial'],
        plot_path=plot_filename
    )

    # Fine-tuning stage
    print("Unfreezing all layers for fine-tuning...")
    trainer.unfreeze_all_layers()
    optimizer_factory.update(optimizer, lr=config['training']['learning_rates']['final_fine_tuning'])

    print("Starting full model fine-tuning...")
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['training']['epochs']['fine_tuning'],
        plot_path=plot_filename
    )

    # Evaluate
    trainer.evaluate(data_loader=test_loader)

if __name__ == "__main__":
    main("config/fine_tuning_config.yaml")
