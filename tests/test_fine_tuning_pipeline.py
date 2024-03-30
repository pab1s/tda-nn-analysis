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
        'pretrained': True,
    },
    'training': {
        'batch_size': 64,
        'num_epochs': 1,
        'learning_rate': 0.0001,
        'freeze_until_layer': "classifier.1.weight",
    }
}

def test_fine_tuning_loop():
    """Test a short fine-tuning loop to ensure pipeline works with BasicTrainer."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(CONFIG_TEST)
    model = get_model(
        CONFIG_TEST['model']['name'],
        CONFIG_TEST['model']['num_classes'],
        CONFIG_TEST['model']['pretrained']
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': CONFIG_TEST['training']['learning_rate']}
    
    trainer = get_trainer(CONFIG_TEST['trainer'], model=model, device=device)
    
    # Simulate fine-tuning process
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        freeze_until_layer=CONFIG_TEST['training'].get('freeze_until_layer')
    )
    trainer.train(
        train_loader=train_loader,
        num_epochs=CONFIG_TEST['training']['num_epochs'],
        verbose=False
    )
    
    trainer.unfreeze_all_layers()
    trainer.build(
         criterion=criterion,
         optimizer_class=optimizer_class,
         optimizer_params={'lr': 0.00001},
         freeze_until_layer=None
     )
    
    trainer.train(
        train_loader=train_loader,
        num_epochs=CONFIG_TEST['training']['num_epochs'],
        verbose=False
    )
    
    accuracy = trainer.evaluate(
        test_loader=test_loader,
        verbose=False
    )

    assert accuracy >= 0, "Accuracy should be non-negative after fine-tuning"
