import torch
import yaml
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from utils.metrics import Accuracy, Precision, Recall, F1Score
from trainers.basic_trainer import BasicTrainer
from factories.model_factory import ModelFactory
from factories.optimizer_factory import OptimizerFactory
from factories.loss_factory import LossFactory
from factories.callback_factory import CallbackFactory
from os import path

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transforms = get_transforms(config['data']['transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    total_size = len(data)
    test_size = int(total_size * config['data']['test_size'])
    val_size = int((total_size - test_size) * config['data']['val_size'])
    train_size = total_size - test_size - val_size

    data_train, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, data_val = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    train_loader = DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(data_val, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(data_test, batch_size=config['training']['batch_size'], shuffle=False)

    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], num_classes=config['model']['parameters']['num_classes'], pretrained=config['model']['parameters']['pretrained']).to(device)

    loss_factory = LossFactory()
    criterion = loss_factory.create(config['training']['loss_function']['type'])

    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create(config['training']['optimizer']['type'])
    optimizer_params = {'lr': config['training']['optimizer']['parameters']['learning_rate']}

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dataset_time = f"{config['model']['type']}_{config['data']['name']}_{current_time}"
    log_filename = path.join(config['paths']['log_path'], f"log_{model_dataset_time}.csv")

    callbacks_config = config['callbacks']
    if "CSVLogging" in callbacks_config:
        callbacks_config["CSVLogging"]["parameters"]["csv_path"] = log_filename

    trainer = BasicTrainer(model=model, device=device)

    metrics = [Accuracy(), Precision(), Recall(), F1Score()]

    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer,
        optimizer_params=optimizer_params,
        metrics=metrics
    )

    callback_factory = CallbackFactory()
    callbacks = []
    for name, params in callbacks_config.items():
        callback = callback_factory.create(name, **params["parameters"])
        callbacks.append(callback)

    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['training']['num_epochs'],
        callbacks=callbacks,
    )
    
    trainer.evaluate(data_loader=test_loader)

if __name__ == "__main__":
    main("config/config.yaml")
