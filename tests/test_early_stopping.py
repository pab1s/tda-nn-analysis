import pytest
from trainers import get_trainer
from callbacks import EarlyStopping
from factories.callback_factory import CallbackFactory
from utils.metrics import Accuracy
from datasets.transformations import get_transforms
from datasets.dataset import get_dataset
from factories.model_factory import ModelFactory
import torch
import yaml

# Load test configuration
with open("./config/config_test.yaml", 'r') as file:
    CONFIG_TEST = yaml.safe_load(file)

@pytest.mark.parametrize("patience,delta,num_epochs", [
    (1, 0.0, 10), # Early stopping should trigger after 2 epochs
])
def test_early_stopping(patience, delta, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations and loading
    transforms = get_transforms(CONFIG_TEST['data']['transforms'])
    data = get_dataset(
        name=CONFIG_TEST['data']['name'],
        root_dir=CONFIG_TEST['data']['dataset_path'],
        train=True,
        transform=transforms
    )

    # Split data into training and testing sets
    train_size = int(0.64 * len(data))
    test_size = len(data) - train_size
    data_train, data_test = torch.utils.data.random_split(data, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=CONFIG_TEST['training']['batch_size'], shuffle=False)

    # Initialize model
    model_factory = ModelFactory()
    model = model_factory.create(CONFIG_TEST['model']['name'], num_classes=CONFIG_TEST['model']['num_classes'], pretrained=CONFIG_TEST['model']['pretrained'])

    # Initialize criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': CONFIG_TEST['training']['learning_rate']}
    metrics = [Accuracy()]

    # Get trainer and build it
    trainer = get_trainer(CONFIG_TEST['trainer'], model=model, device=device)
    
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer_class,
        optimizer_params=optimizer_params,
        metrics=metrics
    )

    # Initialize EarlyStopping callback
    early_stopping_callback = EarlyStopping(patience=patience, verbose=True, monitor='val_loss', delta=delta)
    early_stopping_callback.set_model_and_optimizer(model, trainer.optimizer)

    # Train the model
    trainer.train(
        train_loader=train_loader,
        num_epochs=num_epochs,
        valid_loader=test_loader,
        callbacks=[early_stopping_callback],
    )

    # Assert that early stopping was triggered
    assert early_stopping_callback.early_stop, "Early stopping did not trigger as expected."

# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
