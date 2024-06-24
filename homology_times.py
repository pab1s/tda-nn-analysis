import torch
import yaml
import numpy as np
import scipy.spatial
import time
import logging
from scipy.cluster.hierarchy import linkage
from torch.utils.data import DataLoader, random_split
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from factories.model_factory import ModelFactory
from gtda.homology import VietorisRipsPersistence
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_pretrained_model(model_path, config, device):
    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], num_classes=config['model']['parameters']['num_classes'], pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def load_config(config_path: str):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"The configuration file at {config_path} was not found.")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {exc}")
        raise

def register_all_hooks(model, activations):
    def get_activation(name):
        def hook(model, input, output):
            # Flattening the output for analysis, reshape as required
            activations[name] = output.detach().cpu().numpy().reshape(output.size(0), -1)
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(get_activation(name))

def incremental_processing(test_loader, model, device, activations):
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            model(inputs)  # Triggers hooks and populates activations
            for name, feature_array in activations.items():
                if feature_array is not None:
                    process_feature_layer(name, feature_array)
                    activations[name] = None  # Clear memory after processing

def process_feature_layer(layer_name, feature_array):
    if feature_array.size > 0:
        # Ensure the array is 2-dimensional
        if len(feature_array.shape) == 1:
            feature_array = feature_array.reshape(-1, 1)  # Reshape if it's a single dimension
        elif len(feature_array.shape) > 2:
            feature_array = feature_array.reshape(feature_array.shape[0], -1)  # Flatten multi-dimensional arrays

        logging.info(f"Computing persistence diagrams for layer {layer_name}")
        distance_matrix = scipy.spatial.distance.pdist(feature_array)
        square_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
        persistence_diagram = compute_persistence_diagrams_using_giotto(square_distance_matrix)

        # Sum of persistence diagram values
        diagram_sum = persistence_diagram[:,1].sum()
        logging.info(f"Layer {layer_name}: Persistence Diagram Shape: {persistence_diagram.shape}")
        logging.info(f"Layer {layer_name}: Sum of Persistence Diagram Values: {diagram_sum}")
        logging.info(f"Actual diagram: {persistence_diagram}")
    else:
        logging.info(f"Layer {layer_name}: No features to process.")

def compute_persistence_diagram_using_single_linkage(distance_matrix):
    condensed_matrix = scipy.spatial.distance.squareform(distance_matrix)
    deaths = linkage(condensed_matrix, method='single')[:, 2]
    return np.sort(np.array([[0, d] for d in deaths]))

def compute_persistence_diagrams_using_giotto(distance_matrix, dimensions=[0,1]):
    vr_computator = VietorisRipsPersistence(homology_dimensions=dimensions, metric="precomputed")
    diagrams = vr_computator.fit_transform([distance_matrix])[0]
    # return np.sort(diagrams[diagrams[:, 2] == 0][:, :2])  # Filter zero-dimensional features
    return np.sort(diagrams[:, :2])

def perform_lle_and_plot(features, n_neighbors=10, n_components=2, title="LLE Embedding"):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method='standard')
    transformed_features = lle.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c='blue', marker='o', edgecolor='k')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

def main(config_path: str, model_path: str):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transforms = get_transforms(config['data']['eval_transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    total_size = len(data)
    test_size = int(total_size * config['data']['test_size'])
    _, data_test = random_split(data, [total_size - test_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))    
    test_loader = DataLoader(data_test, batch_size=len(data_test), shuffle=False)  # Reduced batch size to manage memory better
    
    model = load_pretrained_model(model_path, config, device)
    print(model)
    activations = {}
    register_all_hooks(model, activations)
    
    incremental_processing(test_loader, model, device, activations)

if __name__ == "__main__":
    main("config/config_resnet.yaml", "outputs/models/resnet18_CarDataset_SGD_64_2024-05-30_21-44-58.pth")
