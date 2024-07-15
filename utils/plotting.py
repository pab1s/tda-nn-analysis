import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_metrics(metrics_lists, val_metrics_lists=None, keywords=None) -> None:
    """
    Plots the metrics for training and validation data.

    Args:
        metrics_lists (list): A list of lists containing the metrics values for each experiment.
        val_metrics_lists (list, optional): A list of lists containing the validation metrics values for each experiment. Defaults to None.
        keywords (list, optional): A list of keywords to label each experiment. Defaults to None.

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")

    if not keywords:
        keywords = [f"Experiment {i+1}" for i in range(len(metrics_lists))]

    if not any(metrics_lists) and (val_loss_lists is None or not any(val_loss_lists)):
        print("No data available for plotting.")
        return

    # Calculate maximum epoch counts for training data
    train_max_epochs = max((max((len(metrics) for metrics in experiment if metrics), default=0) for experiment in metrics_lists), default=0)

    # If val_metrics_lists is provided and not empty, calculate its max epochs
    if val_metrics_lists and any(val_metrics_lists):
        val_max_epochs = max((max((len(metrics) for metrics in experiment if metrics), default=0) for experiment in val_metrics_lists), default=0)
        max_epochs = max(train_max_epochs, val_max_epochs)
    else:
        max_epochs = train_max_epochs
    
    if max_epochs == 0:
        print("No epochs data available for plotting.")
        return

    plt.figure(figsize=(10, 6))
    epochs_x = np.arange(1, max_epochs + 1)
    colors = sns.color_palette("husl", n_colors=len(metrics_lists))

    # Plotting training metrics
    for idx, experiment_metrics in enumerate(metrics_lists):
        if not experiment_metrics:
            continue

        # Handle training metrics
        metrics_array = np.full((len(experiment_metrics), max_epochs), np.nan)

        for i, metrics in enumerate(experiment_metrics):
            metrics_array[i, :len(metrics)] = metrics

        mean_loss = np.nanmean(metrics_array, axis=0)
        std_loss = np.nanstd(metrics_array, axis=0)

        if not np.isnan(mean_loss).all():
            plt.plot(epochs_x, mean_loss, label=f"{keywords[idx]}", color=colors[idx])
            plt.fill_between(epochs_x, mean_loss - std_loss, mean_loss + std_loss, color=colors[idx], alpha=0.3)

        # Handle validation metrics if provided
        if val_loss_lists and idx < len(val_loss_lists) and val_loss_lists[idx]:
            val_experiment_metrics = val_loss_lists[idx]

            if val_experiment_metrics and any(val_experiment_metrics):
                val_metrics_array = np.full((len(val_experiment_metrics), max_epochs), np.nan)

                for i, metrics in enumerate(val_experiment_metrics):
                    val_metrics_array[i, :len(metrics)] = metrics

                val_mean_loss = np.nanmean(val_metrics_array, axis=0)
                val_std_loss = np.nanstd(val_metrics_array, axis=0)

                if not np.isnan(val_mean_loss).all():
                    plt.plot(epochs_x, val_mean_loss, label=f"Val: {keywords[idx]}", color=colors[idx], linestyle='--')
                    plt.fill_between(epochs_x, val_mean_loss - val_std_loss, val_mean_loss + val_std_loss, color=colors[idx], alpha=0.1)

    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.xticks(epochs_x)
    plt.legend()
    plt.show()

def plot_loss(training_epoch_losses, validation_epoch_losses, plot_path) -> None:
    """
    Plots the training and validation losses over epochs.

    Args:
        training_epoch_losses (list): List of training losses for each epoch.
        validation_epoch_losses (list): List of validation losses for each epoch.
        plot_path (str): Path to save the plot.

    Returns:
        None
    """

    plt.figure(figsize=(10, 5))
    
    epochs = np.arange(1, len(training_epoch_losses) + 1)
    
    plt.plot(epochs, training_epoch_losses, label="Training Loss", marker='o')
    
    if validation_epoch_losses:
        validation_epochs = np.linspace(1, len(training_epoch_losses), num=len(validation_epoch_losses))
        plt.plot(validation_epochs, validation_epoch_losses, label="Validation Loss", marker='x')
    
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

def plot_lr_vs_loss(log_lrs, losses, plot_path) -> None:
    """
    Plots the learning rate vs loss.

    Args:
        log_lrs (list): List of log learning rates.
        losses (list): List of losses.
        plot_path (str): Path to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(log_lrs, losses, label="Learning Rate vs Loss", marker='o')
    
    plt.title("Learning Rate vs Loss")
    plt.xlabel("Log Learning Rate")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

import os

def find_csv_by_keywords(directory, keywords) -> list:
    """
    Finds CSV files in the specified directory that contain the specified keywords in their filenames.

    Args:
        directory (str): The directory path where the CSV files are located.
        keywords (list): A list of keywords to search for in the filenames.

    Returns:
        list: A list of lists containing the DataFrames of the CSV files found for each keyword.
    """

    keyword_files = {keyword: [] for keyword in keywords}
    
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            for keyword in keywords:
                if keyword in file:
                    df = pd.read_csv(os.path.join(directory, file))
                    keyword_files[keyword].append(df)
    
    result = [keyword_files[keyword] for keyword in keywords]
    return result


if __name__ == "__main__":
    directory = 'logs/MM'
    keywords = ['8', '16', '32', '64']
    plot_titles = ['Tamaño de lote 8', 'Tamaño de lote 16', 'Tamaño de lote 32', 'Tamaño de lote 64']
    dataframes_list = find_csv_by_keywords(directory, keywords)

    train_loss_lists = [[df['train_loss'].tolist() for df in dataframes] for dataframes in dataframes_list]
    val_loss_lists = [[df['val_loss'].tolist() for df in dataframes] for dataframes in dataframes_list]
    train_precision_lists = [[df['train_precision'].tolist() for df in dataframes] for dataframes in dataframes_list]
    val_precision_lists = [[df['val_precision'].tolist() for df in dataframes] for dataframes in dataframes_list]
    val_accuracy_lists = [[df['val_accuracy'].tolist() for df in dataframes] for dataframes in dataframes_list]
    val_f1_score_lists = [[df['val_f1_score'].tolist() for df in dataframes] for dataframes in dataframes_list]

    plot_metrics(train_loss_lists, val_loss_lists, plot_titles)
    plot_metrics(train_precision_lists, val_precision_lists, plot_titles)
    plot_metrics(val_accuracy_lists, val_f1_score_lists, plot_titles)
