import os
import argparse
import torch
import yaml
import numpy as np
import scipy.spatial
import datetime
import logging
import time
from torch.utils.data import DataLoader, random_split, Subset
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from factories.model_factory import ModelFactory
from gtda.homology import VietorisRipsPersistence

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pretrained_model(model_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """
    Load a pretrained model from a specified path using configurations.

    Args:
        model_path (str): Path to the model file.
        config (dict): Configuration dictionary specifying model details.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """

    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], num_classes=config['model']['parameters']['num_classes'], pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")

def register_all_hooks(model: torch.nn.Module, activations: dict, layer_progress: dict) -> None:
    """
    Register forward hooks to capture output activations of specific layers during model forwarding.

    Args:
        model (torch.nn.Module): The model from which to capture activations.
        activations (dict): A dictionary to store the activations.
        layer_progress (dict): A dictionary to track the progress of output capturing.
    """

    relevant_layers = [name for name, layer in model.named_modules() if isinstance(layer, (torch.nn.ReLU, torch.nn.SiLU, torch.nn.Linear))]
    total_layers = len(relevant_layers)

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy().reshape(output.size(0), -1)
            current_layer_index = relevant_layers.index(name)
            progress = (current_layer_index + 1) / total_layers * 100
            layer_progress[name] = progress
        return hook

    for name in relevant_layers:
        layer = dict(model.named_modules())[name]
        layer.register_forward_hook(get_activation(name))

def compute_persistence_diagrams_using_giotto(distance_matrix: np.ndarray, dimensions: list = [0, 1]) -> np.ndarray:
    """
    Compute persistence diagrams using Vietoris-Rips complex from a precomputed distance matrix.

    Args:
        distance_matrix (np.ndarray): A square matrix of pairwise distances.
        dimensions (list): List of homology dimensions to compute.

    Returns:
        np.ndarray: Array of persistence diagrams.
    """

    vr_computator = VietorisRipsPersistence(homology_dimensions=dimensions, metric="precomputed")
    diagrams = vr_computator.fit_transform([distance_matrix])[0]
    return np.sort(diagrams[:, :2])

def save_persistence_diagram(persistence_diagram: np.ndarray, layer_name: str, dataset_type: str, model_name: str, progress: float, persistence_dir: str) -> None:
    """
    Save the computed persistence diagram to a text file.

    Args:
        persistence_diagram (np.ndarray): Array of persistence intervals.
        layer_name (str): Name of the layer for which the diagram was computed.
        dataset_type (str): Type of the dataset (e.g., train, test).
        model_name (str): Name of the model.
        progress (float): Percentage of the progress in the model processing.
        persistence_dir (str): Directory to save the persistence diagrams.
    """

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"{model_name}_{layer_name}_{dataset_type}_{progress:.2f}percent_{timestamp}.txt"
    dataset_persistence_dir = os.path.join(persistence_dir, dataset_type)
    os.makedirs(dataset_persistence_dir, exist_ok=True)
    filepath = os.path.join(dataset_persistence_dir, filename)
    
    with open(filepath, 'w') as f:
        for birth, death in persistence_diagram:
            f.write(f'{birth} {death}\n')
    
    logging.info(f'Saved persistence diagram to {filepath}')

def incremental_processing(loader: DataLoader, model: torch.nn.Module, device: torch.device, activations: dict, dataset_type: str, model_name: str, layer_progress: dict, persistence_dir: str) -> None:
    """
    Process data incrementally, computing persistence diagrams for model activations layer by layer.

    Args:
        loader (DataLoader): DataLoader for the dataset.
        model (torch.nn.Module): Pretrained model.
        device (torch.device): Device on which computation is performed.
        activations (dict): Dictionary holding activations.
        dataset_type (str): Type of the dataset (e.g., 'train', 'valid').
        model_name (str): Name of the model.
        layer_progress (dict): Progress tracking for each layer.
        persistence_dir (str): Directory to save persistence diagrams.
    """

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            output = model(inputs)
            for name, feature_array in activations.items():
                if feature_array is not None:
                    progress = layer_progress.get(name, 0)
                    process_feature_layer(name, feature_array, dataset_type, model_name, progress, persistence_dir)
                    activations[name] = None

def process_feature_layer(layer_name: str, feature_array: np.ndarray, dataset_type: str, model_name: str, progress: float, persistence_dir: str) -> None:
    """
    Process a single layer's features to compute and save its persistence diagram.

    Args:
        layer_name (str): Name of the layer.
        feature_array (np.ndarray): Activations of the layer.
        dataset_type (str): Type of the dataset being processed.
        model_name (str): Model identifier.
        progress (float): Progress of processing through the model.
        persistence_dir (str): Directory where diagrams should be saved.
    """

    if feature_array.size > 0:
        feature_array = feature_array.reshape(-1, feature_array.shape[-1])
        logging.info(f"Computing persistence diagrams for layer {layer_name} at {progress:.2f}% through the model")
        distance_matrix = scipy.spatial.distance.pdist(feature_array)
        square_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
        persistence_diagram = compute_persistence_diagrams_using_giotto(square_distance_matrix)
        save_persistence_diagram(persistence_diagram, layer_name, dataset_type, model_name, progress, persistence_dir)
    else:
        logging.info(f"Layer {layer_name}: No features to process at {progress:.2f}% through the model")

def process_dataset(loader, dataset_type, model_name, persistence_dir) -> None:
    """
    Process a dataset to compute persistence diagrams.

    Args:
        loader (DataLoader): DataLoader for the dataset.
        dataset_type (str): Type of the dataset (e.g., 'train', 'valid').
        model_name (str): Name of the model.
        persistence_dir (str): Directory to save persistence diagrams.
    """
    
    for i, (inputs, labels) in enumerate(loader):
        feature_array = inputs.view(inputs.size(0), -1).numpy()
        logging.info(f"Computing persistence diagrams for {dataset_type} set, batch {i}")
        distance_matrix = scipy.spatial.distance.pdist(feature_array)
        square_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
        persistence_diagram = compute_persistence_diagrams_using_giotto(square_distance_matrix)
        save_persistence_diagram(persistence_diagram, f'batch_{i}', dataset_type, model_name, 100.0 * (i + 1) / len(loader), persistence_dir)


def main(config_name: str, model_name: str) -> None:
    """
    Main function for computing persistence diagrams from a pretrained model.

    Args:
        config_name (str): Name of the configuration file.
        model_name (str): Name of the pretrained model.
    """

    model_path = f"outputs/models/DENSENET_REGULARIZADOR/{model_name}.pth"
    config = load_config(f"config/{config_name}.yaml")

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Running on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(model_path, config, device)

    transforms = get_transforms(config['data']['eval_transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    # Split data
    total_size = len(data)
    test_size = int(total_size * config['data']['test_size'])
    val_size = int(total_size * config['data']['val_size'])
    train_size = total_size - test_size - val_size
    assert train_size > 0 and val_size > 0 and test_size > 0, "One of the splits has zero or negative size."
    data_train, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))
    data_train, data_val = random_split(data_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['random_seed']))

    # Truncate each set to 128 images
    num_images = 128
    data_train = Subset(data_train, range(min(num_images, len(data_train))))
    data_val = Subset(data_val, range(min(num_images, len(data_val))))
    data_test = Subset(data_test, range(min(num_images, len(data_test))))

    train_loader = DataLoader(data_train, batch_size=num_images, shuffle=True)
    valid_loader = DataLoader(data_val, batch_size=num_images, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=num_images, shuffle=False)

    model_dir = os.path.join("output_files", model_name)
    os.makedirs(model_dir, exist_ok=True)
    persistence_dir = os.path.join(model_dir, 'persistence_diagrams')
    lle_dir = os.path.join(model_dir, 'lle_plots')
    os.makedirs(persistence_dir, exist_ok=True)
    os.makedirs(lle_dir, exist_ok=True)

    activations = {}
    layer_progress = {}
    register_all_hooks(model, activations, layer_progress)

    loaders = [train_loader, valid_loader, test_loader]
    dataset_types = ['train', 'valid', 'test']

    for loader, dataset_type in zip(loaders, dataset_types):
        time_start = time.time()
        # process_dataset(loader, dataset_type, model_name, persistence_dir)
        incremental_processing(loader, model, device, activations, dataset_type, model_name, layer_progress, persistence_dir, lle_dir)
        time_end = time.time()
        logging.info(f"{dataset_type.capitalize()} processing time: {time_end - time_start:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute persistence diagrams from a pretrained model.")
    parser.add_argument("config_name", type=str, help="Name of the configuration file.")
    parser.add_argument("model_name", type=str, help="Name of the pretrained model.")
    args = parser.parse_args()

    main(args.config_name, args.model_name)

