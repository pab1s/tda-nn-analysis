import pytest
from trainers import get_trainer
from utils.data_utils import get_dataloaders
from models import get_model
import torch

CONFIG_TEST = {
    'trainer': 'BasicTrainer',
    'data': {
        'name': 'CIFAR10',
        'dataset_path': './data',
    },
    'model': {
        'name': 'efficientnet_b0',
        'num_classes': 10,
        'pretrained': False,
    },
    'training': {
        'batch_size': 64,
        'num_epochs': 1,
        'learning_rate': 0.001,
    },
    'paths': {
        'log_path': "./logs/log_test.csv",
        'plot_path': "./outputs/figures/plot_test.png",
    }
}

def test_training_loop():
    """Test a short training loop to ensure pipeline works with BasicTrainer."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(CONFIG_TEST)
    model = get_model(
        CONFIG_TEST['model']['name'],
        CONFIG_TEST['model']['num_classes'],
        CONFIG_TEST['model']['pretrained']
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG_TEST['training']['learning_rate']
    )

    trainer = get_trainer(CONFIG_TEST['trainer'], model=model, device=device)
    
    trainer.build(
        criterion=criterion,
        optimizer=optimizer,
    )
    trainer.train(
        train_loader=train_loader,
        num_epochs=CONFIG_TEST['training']['num_epochs'],
        log_path=CONFIG_TEST['paths']['log_path'],
        plot_path=CONFIG_TEST['paths']['plot_path'],
        verbose=False
    )
    accuracy = trainer.evaluate(
        test_loader=test_loader,
        verbose=False
    )

    assert accuracy >= 0, "Accuracy should be non-negative"
