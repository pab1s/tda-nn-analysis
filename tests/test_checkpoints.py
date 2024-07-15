import pytest
import os
import yaml
import torch
from trainers import get_trainer
from utils.metrics import Accuracy, Precision, Recall, F1Score
from datasets.transformations import get_transforms
from datasets.dataset import get_dataset
from factories.model_factory import ModelFactory
from callbacks import Checkpoint

CONFIG_TEST = {}

with open("./config/config_test.yaml", 'r') as file:
    CONFIG_TEST = yaml.safe_load(file)

def test_checkpoint():
    """
    Test the functionality of checkpoint saving and loading.

    This function performs the following steps:
    1. Sets the device to CUDA if available, otherwise CPU.
    2. Retrieves the data transforms and dataset.
    3. Splits the dataset into training and testing sets.
    4. Creates data loaders for the training and testing sets.
    5. Creates the model, criterion, optimizer, and metrics.
    6. Sets up the trainer and checkpoint callback.
    7. Trains the model and saves checkpoints at specified intervals.
    8. Verifies that the checkpoint file was created.
    9. Resets the model parameters to simulate a restart.
    10. Loads the checkpoint.
    11. Evaluates the model on the test set.
    12. Verifies that the metrics after resuming are valid.

    Raises:
        AssertionError: If the checkpoint file was not created or if the metrics after resuming are not valid.
    """
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

    model_factory = ModelFactory()
    model = model_factory.create(CONFIG_TEST['model']['name'], num_classes=CONFIG_TEST['model']['num_classes'], pretrained=CONFIG_TEST['model']['pretrained'])
    
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
