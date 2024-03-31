import torch
import yaml
from datetime import datetime
from utils.metrics import Accuracy, Precision
from utils.data_utils import get_dataloaders
from models import get_model
from trainers import get_trainer
from os import path


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(config)

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
    log_filename = path.join(
        config['paths']['log_path'], f"log_finetuning_{model_dataset_time}.csv")
    plot_filename = path.join(
        config['paths']['plot_path'], f"plot_finetuning_{model_dataset_time}.png")

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
        num_epochs=config['training']['fine_tuning_epochs'],
        log_path=log_filename,
        plot_path=plot_filename
    )

    trainer.evaluate(test_loader=test_loader)


if __name__ == "__main__":
    main("config/fine_tuning_config.yaml")
