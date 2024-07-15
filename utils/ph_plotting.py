import os
import re
import gudhi as gd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def read_persistence_intervals(filename) -> List[Tuple[float, float]]:
    """
    Reads persistence intervals from a file.
    
    Parameters:
        filename (str): The path to the file containing persistence intervals.
    
    Returns:
        List[Tuple[float, float]]: A list of tuples containing the birth and death values.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        intervals = [tuple(map(float, line.strip().split())) for line in lines]
    return intervals

def extract_percentage(filename: str) -> Optional[float]:
    """
    Extract the percentage from the filename using regex.
    
    Args:
        filename (str): The filename containing the percentage.

    Returns:
        Optional[float]: The extracted percentage or None if not found.
    """

    match = re.search(r'_([\d\.]+)percent_', filename)

    return float(match.group(1)) if match else None

def extract_info(directory_name: str, choice: str) -> Optional[int]:
    """
    Extract information based on a choice from the directory name using regex.

    Args:
        directory_name (str): Name of the directory.
        choice (str): Choice from ['batch_size', 'optimizer', 'architecture'].

    Returns:
        Optional[int]: Indexed value based on the directory pattern.
    """

    patterns = {
        'batch_size': r'_(\d+)_\d{4}-\d{2}-\d{2}',
        'optimizer': r'_(Adam|SGD)_',
        'architecture': r'^(densenet121|efficientnet_b0|resnet18)_'
    }
    indexes = {
        'batch_size': {'8': 0, '16': 1, '32': 2, '64': 3},
        'optimizer': {'Adam': 0, 'SGD': 1},
        'architecture': {'densenet121': 0, 'efficientnet_b0': 1, 'resnet18': 2}
    }

    if choice in patterns:
        match = re.search(patterns[choice], directory_name)
        if match:
            return indexes[choice][match.group(1)]
    return None

def compute_total_persistence(persistence_intervals: List[Tuple[float, float]]) -> float:
    """
    Calculate the total persistence from a list of persistence intervals.

    Args:
        persistence_intervals (List[Tuple[float, float]]): List of intervals.

    Returns:
        float: The total persistence calculated from the intervals.
    """

    max_lifetime = max(death - birth for birth, death in persistence_intervals if death != np.inf)
    return sum((death if death != np.inf else max_lifetime) - birth for birth, death in persistence_intervals) # / max_lifetime

def plot_barcode_sets(base_path: str, model_dir: str, datasets: List[str]) -> None:
    """
    Plot barcode values for different datasets within a model directory.

    Args:
        base_path (str): Base directory where barcodes by dataset are located.
        model_dir (str): Specific model directory to plot.
        datasets (List[str]): List of datasets to process.

    Returns:
        None: This function plots a graph.
    """

    colors = sns.color_palette("hls", len(datasets))
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    for idx, dataset in enumerate(datasets):
        dataset_path = os.path.join(base_path, model_dir, 'persistence_diagrams', dataset)
        barcode_values = []
        if os.path.isdir(dataset_path):
            for filename in sorted(os.listdir(dataset_path)):
                if filename.endswith(".txt"):
                    percentage = extract_percentage(filename)
                    if percentage is not None:
                        filepath = os.path.join(dataset_path, filename)
                        intervals = read_persistence_intervals(filepath)
                        barcode = compute_total_persistence(intervals)
                        barcode_values.append((percentage, barcode))

            if barcode_values:
                barcode_values.sort()
                percentages, entropies = zip(*barcode_values)
                sns.regplot(x=np.array(percentages), y=np.array(entropies), order=2,
                            scatter_kws={'s': 100, 'color': colors[idx], 'alpha': 0.5},
                            line_kws={'color': colors[idx], 'lw': 6},
                            label=f'{dataset.capitalize()} Set')

    plt.ylim(0, 250000)
    plt.xlabel('Percentage', fontsize=20)
    plt.ylabel('Total Persistence', fontsize=20)
    plt.legend(prop={'size': 24})
    plt.tight_layout()
    plt.show()

def plot_barcodes_groups(base_directory_path: str, choice: str) -> None:
    """
    Plot barcode values grouped by specified attributes like batch size, optimizer, or architecture.

    Args:
        base_directory_path (str): Base directory containing the model directories.
        choice (str): Attribute to group the barcodes by.
    """

    colors = sns.color_palette("hls", 4)
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    for model_dir in os.listdir(base_directory_path):
        model_path = os.path.join(base_directory_path, model_dir)
        if os.path.isdir(model_path):
            index = extract_info(model_dir, choice)
            if index is not None:
                persistence_diagram_path = os.path.join(model_path, 'persistence_diagrams/test')
                if os.path.exists(persistence_diagram_path):
                    files_with_percentages = [(f, extract_percentage(f)) for f in os.listdir(persistence_diagram_path) if f.endswith(".txt") and extract_percentage(f) is not None]
                    sorted_files = sorted(files_with_percentages, key=lambda x: x[1])
                    percentages, barcodes = zip(*[(p, compute_total_persistence(read_persistence_intervals(os.path.join(persistence_diagram_path, f)))) for f, p in sorted_files])
                    sns.regplot(x=np.array(percentages), y=np.array(barcodes), order=2,
                                scatter_kws={'s': 100, 'color': colors[index], 'alpha': 0.5},
                                line_kws={'color': colors[index], 'lw': 6},
                                label=f'{model_dir}')

    plt.ylim(0, 250000)
    plt.xlabel('Percentage', fontsize=20)
    plt.ylabel('Total Persistence', fontsize=20)
    plt.legend(prop={'size': 24})
    plt.tight_layout()
    plt.show()

def barcode_plot(filename) -> None:
    """
    Plots the persistence barcode for a given file containing persistence intervals.

    Parameters:
    filename (str): The path to the file containing persistence intervals.

    Returns:
    None
    """
    persistence_intervals = read_persistence_intervals(filename)

    # Separate persistence intervals by type
    H0_intervals = [(0, interval) for interval in persistence_intervals if interval[0] == 0.0]
    H1_intervals = [(1, interval) for interval in persistence_intervals if interval[0] != 0.0]

    diag = H0_intervals + H1_intervals

    # Visualize persistence with GUDHI
    plt.figure(figsize=(12, 8))
    gd.plot_persistence_barcode(diag, legend=True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.title('EfficientNet-B0 Regularizado 100%')
    plt.show()

if __name__ == '__main__':
    filename = 'output_files/FINAL/m_efficientnet_base/efficientnet_b0_Adam_8_2024-06-01_14-51-19/efficientnet_b0_Adam_8_2024-06-01_14-51-19_100percent_train.txt'
    barcode_plot(filename)
