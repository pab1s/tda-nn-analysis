import pytest
from trainers import get_trainer
from utils.metrics import Accuracy, Precision, Recall, F1Score
from datasets.transformations import get_transforms
from datasets.dataset import get_dataset
from models import get_model
import torch
import yaml

CONFIG_TEST = {}

with open("./config/config_test.yaml", 'r') as file:
    CONFIG_TEST = yaml.safe_load(file)

def test_fine_tuning_loop():
    """Test a short fine-tuning loop to ensure pipeline works with BasicTrainer."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transforms = get_transforms(CONFIG_TEST)

    data = get_dataset(
        name=CONFIG_TEST['data']['name'],
        root_dir=CONFIG_TEST['data']['dataset_path'],
        train=True,
        transform=transforms
    )

    train_size = int((1 - 0.2) * len(data))
    test_size = len(data) - train_size
    train_size = int(train_size * (1 - 0.2))
    val_size = len(data) - train_size - test_size

    data_train, data_test = torch.utils.data.random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(42))
    data_train, data_val = torch.utils.data.random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(data_val, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=False)

    model = get_model(
        CONFIG_TEST['model']['name'],
        CONFIG_TEST['model']['num_classes'],
        CONFIG_TEST['model']['pretrained']
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': CONFIG_TEST['training']['learning_rate']}
    metrics = [Accuracy(), Precision(), Recall(), F1Score()]
    
    trainer = get_trainer(CONFIG_TEST['trainer'], model=model, device=device)
    
    # Simulate fine-tuning process
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        freeze_until_layer=CONFIG_TEST['training'].get('freeze_until_layer'),
        metrics=metrics
    )
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=CONFIG_TEST['training']['num_epochs'],
        verbose=False
    )
    
    trainer.unfreeze_all_layers()
    trainer.build(
         criterion=criterion,
         optimizer_class=optimizer_class,
         optimizer_params={'lr': 0.00001},
         freeze_until_layer=None,
         metrics=metrics
     )
    
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=CONFIG_TEST['training']['num_epochs'],
        verbose=False
    )
    
    _, metrics_results = trainer.evaluate(
        data_loader=test_loader,
        verbose=False
    )

    assert len(metrics_results) == len(metrics)
    assert all([v >= 0 for v in metrics_results.values()])
