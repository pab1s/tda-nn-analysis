import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from utils.metrics import Accuracy, Precision, Recall, F1Score
from models import get_model
from trainers import get_trainer
from os import path


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transforms = get_transforms(config)
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    train_size = int((1 - config['data']['test_size']) * len(data))
    test_size = len(data) - train_size
    train_size = int(train_size * (1 - config['data']['val_size']))
    val_size = len(data) - train_size - test_size

    data_train, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, data_val = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    train_loader = DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(data_val, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(data_test, batch_size=config['training']['batch_size'], shuffle=False)

    model = get_model(
        config['model']['name'],
        config['model']['num_classes'],
        pretrained=True
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': config['training']['learning_rate']}
    metrics = [Accuracy(), Precision()]

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dataset_time = f"{config['model']['name']}_{config['data']['name']}_{current_time}"
    log_filename = path.join(config['paths']['log_path'], f"log_finetuning_{model_dataset_time}.csv")
    plot_filename = path.join(config['paths']['plot_path'], f"plot_finetuning_{model_dataset_time}.png")

    trainer = get_trainer(config['trainer'], model=model, device=device)

    # Stage 1: Train with some layers frozen
    print("Starting initial training stage with frozen layers...")
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        freeze_until_layer=config['training'].get('freeze_until_layer'),
        metrics=metrics
    )
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['training']['initial_epochs'],
        log_path=log_filename,
        plot_path=plot_filename
    )

    # Stage 2: Unfreeze all layers for full fine-tuning
    print("Unfreezing all layers for fine-tuning...")
    trainer.unfreeze_all_layers()

    optimizer_params['lr'] = config['training']['final_fine_tuning_learning_rate']
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        metrics=metrics
    )

    print("Starting full model fine-tuning...")
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['training']['fine_tuning_epochs'],
        log_path=log_filename,
        plot_path=plot_filename
    )

    trainer.evaluate(data_loader=test_loader)


if __name__ == "__main__":
    main("config/fine_tuning_config.yaml")
