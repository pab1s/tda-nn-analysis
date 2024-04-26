import pytest
from trainers import get_trainer
from callbacks import EarlyStopping
from utils.metrics import Accuracy
from datasets.transformations import get_transforms
from datasets.dataset import get_dataset
from models import get_model
import torch
import yaml

CONFIG_TEST = {}

with open("./config/config_test.yaml", 'r') as file:
    CONFIG_TEST = yaml.safe_load(file)

def test_early_stopping():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = get_transforms(CONFIG_TEST['data']['transforms'])
    data = get_dataset(
        name=CONFIG_TEST['data']['name'],
        root_dir=CONFIG_TEST['data']['dataset_path'],
        train=True,
        transform=transforms
    )

    # Use a NaiveTrainer to test the early stopping
    CONFIG_TEST['trainer'] = 'NaiveTrainer'

    train_size = int(0.64 * len(data))
    test_size = len(data) - train_size
    data_train, data_test = torch.utils.data.random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=False)

    model = get_model(CONFIG_TEST['model']['name'], CONFIG_TEST['model']['num_classes'], CONFIG_TEST['model']['pretrained']).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    optimizer_params = {'lr': CONFIG_TEST['training']['learning_rate']}
    metrics = [Accuracy()]

    trainer = get_trainer(CONFIG_TEST['trainer'], model=model, device=device)
    
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer,
        optimizer_params=optimizer_params,
        metrics=metrics
    )

    early_stopping_callback = EarlyStopping(patience=2, verbose=True, monitor='val_loss', delta=0.1)
    trainer.train(
        train_loader=train_loader,
        num_epochs=3,  # Intentionally, one more epoch than patience as early stopping should trigger
        valid_loader=test_loader,
        callbacks=[early_stopping_callback],
    )

    assert early_stopping_callback.early_stop, "Early stopping did not trigger as expected."

test_early_stopping()
