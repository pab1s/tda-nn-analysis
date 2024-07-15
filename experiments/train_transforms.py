import torch
import yaml
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from utils.metrics import Accuracy, Precision, Recall, F1Score
from factories.model_factory import ModelFactory
from factories.loss_factory import LossFactory
from factories.optimizer_factory import OptimizerFactory
from factories.callback_factory import CallbackFactory
from trainers import get_trainer
from os import path

def main(config_path, transform_type):
    """
    Main function for training and evaluating a model.

    Args:
        config_path (str): The path to the configuration file.
        transform_type (str): The type of transformation to apply to the data.

    Returns:
        None
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # If CUDA not available, finish execution
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit()
    device = torch.device("cuda")
    
    # Load and transform data
    transforms = get_transforms(config['data']['transforms'])
    eval_transforms = get_transforms(config['data']['eval_transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    # Split data
    total_size = len(data)
    test_size = int(total_size * config['data']['test_size'])
    val_size = int(total_size * config['data']['val_size'])
    train_size = total_size - test_size - val_size

    data_train, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, data_val = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    # Apply evaluation transforms to validation and test datasets
    data_test.dataset.transform = eval_transforms
    data_val.dataset.transform = eval_transforms

    # Data loaders using the given batch_size
    train_loader = DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(data_val, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(data_test, batch_size=config['training']['batch_size'], shuffle=False)

    # Model setup
    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], num_classes=config['model']['parameters']['num_classes'], pretrained=config['model']['parameters']['pretrained']).to(device)
    print(model)

    # Loss setup
    loss_factory = LossFactory()
    criterion = loss_factory.create(config['training']['loss_function']['type'])

    # Optimizer setup with given parameters
    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create(config['training']['optimizer']['type'])
    print("Using optimizer: ", optimizer, " with params: ", config['training']['optimizer']['parameters'])

    # Training stages setup
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dataset_time = f"{config['model']['type']}_{config['data']['name']}_{transform_type}_{current_time}"
    log_filename = path.join(config['paths']['log_path'], f"log_finetuning_{model_dataset_time}.csv")

    # Callbacks setup
    callbacks_config = config['callbacks']
    if "CSVLogging" in callbacks_config:
        callbacks_config["CSVLogging"]["parameters"]["csv_path"] = log_filename

    # Metrics and trainer setup
    metrics = [Accuracy(), Precision(), Recall(), F1Score()]
    trainer = get_trainer(config['trainer'], model=model, device=device)

    # Initial training stage
    print("Starting initial training stage with frozen layers...")
    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer,
        optimizer_params=config['training']['optimizer']['parameters'],
        # freeze_until_layer=config['training']['freeze_until_layer'],
        metrics=metrics
    )

    callback_factory = CallbackFactory()
    callbacks = []
    for name, params in callbacks_config.items():
        if name == "Checkpoint":
            params["parameters"]["checkpoint_dir"] = path.join(config['paths']['checkpoint_path'], model_dataset_time)
            params["parameters"]["model"] = model
            params["parameters"]["optimizer"] = trainer.optimizer
            params["parameters"]["scheduler"] = trainer.scheduler
            
        callback = callback_factory.create(name, **params["parameters"])

        if name == "EarlyStopping":
            callback.set_model_and_optimizer(model, trainer.optimizer)

        callbacks.append(callback)

    #trainer.train(
    #    train_loader=train_loader,
    #    valid_loader=valid_loader,
    #    num_epochs=config['training']['epochs']['initial'],
    #    callbacks=callbacks
    #)

    # Fine-tuning stage with all layers unfrozen
    print("Unfreezing all layers for fine-tuning...")
    trainer.unfreeze_all_layers()

    # optimizer_instance = trainer.optimizer
    # optimizer_factory.update(optimizer_instance, config['training']['learning_rates']['initial'])

    print("Starting full model fine-tuning...")
    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['training']['epochs']['fine_tuning'],
        callbacks=callbacks
    )

    # Save model
    model_path = path.join(config['paths']['model_path'], f"{model_dataset_time}.pth")
    torch.save(model.state_dict(), model_path)

    # Evaluate
    trainer.evaluate(data_loader=test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config file.")
    args = parser.parse_args()
    config_path = f"config/{args.config_path}"
    transform_type = path.basename(config_path).split('.')[0]
    
    main(config_path, transform_type)
