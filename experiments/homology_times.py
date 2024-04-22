import torch
import yaml
import numpy as np
import scipy.spatial
import time
import logging
from scipy.cluster.hierarchy import linkage
from torch.utils.data import DataLoader
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from models import get_model
from gtda.homology import VietorisRipsPersistence

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_model(config, device):
    model = get_model(
        config['model']['name'],
        config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    model.classifier = torch.nn.Identity()  # Remove the classifier head
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

def compute_features_and_labels(test_loader, model, device):
    all_features, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            features = model(inputs)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

def compute_persistence_diagram_using_single_linkage(distance_matrix):
    condensed_matrix = scipy.spatial.distance.squareform(distance_matrix)
    deaths = linkage(condensed_matrix, method='single')[:, 2]
    return np.array([[0, d] for d in deaths])

def compute_persistence_diagrams_using_giotto(distance_matrix, dimensions=[0,1]):
    vr_computator = VietorisRipsPersistence(homology_dimensions=dimensions, metric="precomputed")
    diagrams = vr_computator.fit_transform([distance_matrix])[0]
    return diagrams[diagrams[:, 2] == 0][:, :2]  # Filter zero-dimensional features

def main(config_path: str):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transforms = get_transforms(config)
    data_test = get_dataset(config['data']['name'], config['data']['dataset_path'], train=False, transform=transforms)
    test_loader = DataLoader(data_test, batch_size=config['training']['batch_size'], shuffle=False)
    
    model = prepare_model(config, device)
    
    features, labels = compute_features_and_labels(test_loader, model, device)
    logging.info(f"Features shape: {features.shape}")
    
    start_time = time.time()
    distance_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(features))
    logging.info(f"Time taken to compute distance matrix: {time.time() - start_time:.2f}s")
    
    start_time = time.time()
    persistence_diagram_sl = compute_persistence_diagram_using_single_linkage(distance_matrix)
    logging.info(f"Time taken for single linkage: {time.time() - start_time:.2f}s")
    logging.info(f"Persistence Diagram (SL) Shape: {persistence_diagram_sl.shape}")
    
    dims = [0, 1]
    start_time = time.time()
    persistence_diagram_giotto = compute_persistence_diagrams_using_giotto(distance_matrix, dims)
    logging.info(f"Time taken for Giotto: {time.time() - start_time:.2f}s")
    logging.info(f"Persistence Diagram (Giotto) Shape: {persistence_diagram_giotto.shape}")

if __name__ == "__main__":
    main("config/config.yaml")
