import pytest
import os
import yaml
import torch
from trainers import get_trainer
from utils.metrics import Accuracy, Precision, Recall, F1Score
from datasets.transformations import get_transforms
from datasets.dataset import get_dataset
from models import get_model
from callbacks import Checkpoint

CONFIG_TEST = {}

with open("./config/config_test.yaml", 'r') as file:
    CONFIG_TEST = yaml.safe_load(file)

def test_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = get_transforms(CONFIG_TEST['data']['transforms'])
    data = get_dataset(
        name=CONFIG_TEST['data']['name'],
        root_dir=CONFIG_TEST['data']['dataset_path'],
        train=True,
        transform=transforms
    )

    train_size = int(0.64 * len(data))
    test_size = len(data) - train_size
    data_train, data_test = torch.utils.data.random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=False)

    model = get_model(CONFIG_TEST['model']['name'], CONFIG_TEST['model']['num_classes'], CONFIG_TEST['model']['pretrained']).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG_TEST['training']['learning_rate'])
    metrics = [Accuracy(), Precision(), Recall(), F1Score()]

    trainer = get_trainer(CONFIG_TEST['trainer'], model=model, device=device)
    
    checkpoint_dir = "./outputs/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = Checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        save_freq=5,
        verbose=False
    )

    trainer.build(
        criterion=criterion,
        optimizer_class=torch.optim.Adam,
        optimizer_params={'lr': CONFIG_TEST['training']['learning_rate']},
        metrics=metrics
    )
    
    # Train the model and automatically save the checkpoint at the specified interval
    trainer.train(
        train_loader=train_loader,
        num_epochs=6,
        valid_loader=None,
        callbacks=[checkpoint_callback]
    )

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_5.pth')
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    # Zero out the model parameters to simulate a restart
    for param in model.parameters():
        param.data.zero_()

    # Load the checkpoint
    trainer.load_checkpoint(checkpoint_path)

    # Continue training or perform evaluation
    _, metrics_results = trainer.evaluate(test_loader, verbose=False)
    assert all([v >= 0 for v in metrics_results.values()]), "Metrics after resuming are not valid."

test_checkpoint()
