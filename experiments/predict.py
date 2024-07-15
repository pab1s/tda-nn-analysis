import torch
import yaml
import argparse
from torch.utils.data import DataLoader, random_split
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from utils.metrics import Accuracy, Precision, Recall, F1Score
from factories.model_factory import ModelFactory
from factories.optimizer_factory import OptimizerFactory
from factories.loss_factory import LossFactory
from trainers import get_trainer

def evaluate_model(config_path, model_path):
    """
    Evaluate the model using the given configuration and model paths.

    Args:
        config_path (str): The path to the configuration file.
        model_path (str): The path to the model file.

    Returns:
        float: The average loss.
        dict: A dictionary containing the metric results.
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
    assert train_size > 0 and val_size > 0 and test_size > 0, "One of the splits has zero or negative size."
    data_train, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, data_val = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    # Apply evaluation transforms to validation and test datasets
    data_test.dataset.transform = eval_transforms
    data_val.dataset.transform = eval_transforms

    # Data loaders using the given batch_size
    test_loader = DataLoader(data_test, batch_size=config['training']['batch_size'], shuffle=False)

    # Model setup
    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], num_classes=config['model']['parameters']['num_classes'], pretrained=config['model']['parameters']['pretrained']).to(device)
    print(model)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Metrics and trainer setup
    loss_factory = LossFactory()
    criterion = loss_factory.create(config['training']['loss_function']['type'])

    optimizer_factory = OptimizerFactory()
    optimizer = optimizer_factory.create(config['training']['optimizer']['type'])
    optimizer_params = {'lr': config['training']['optimizer']['parameters']['learning_rate']}

    metrics = [Accuracy(), Precision(), Recall(), F1Score()]
    trainer = get_trainer(config['trainer'], model=model, device=device)

    trainer.build(
        criterion=criterion,
        optimizer_class=optimizer,
        optimizer_params=optimizer_params,
        metrics=metrics
    )

    # Evaluate
    avg_loss, metric_results = trainer.evaluate(data_loader=test_loader, metrics=metrics)
    
    print(f"Evaluation - Loss: {avg_loss:.4f}")
    for metric_name, metric_value in metric_results.items():
        print(f"{metric_name}: {metric_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the test set.')
    parser.add_argument('config_filename', type=str, help='Filename of the configuration file within the "config" directory')
    parser.add_argument('model_path', type=str, help='Path to the trained model file (.pth)')

    args = parser.parse_args()

    config_path = f"config/{args.config_filename}"

    evaluate_model(config_path, args.model_path)
