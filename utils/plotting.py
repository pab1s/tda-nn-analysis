import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_loss_statistics_with_seaborn(loss_lists):
    """
    Plots the mean loss per epoch and the standard deviation as a shaded area for multiple experiments with variable epoch lengths using seaborn for improved aesthetics.
    
    Args:
        loss_lists (list of list of lists): Each element is a list of lists where each sublist represents 
                                            the loss per epoch for a single experiment, not necessarily all the same length.
    """
    # Setting up seaborn for better aesthetics
    sns.set(style="whitegrid")
    
    # Determine the maximum number of epochs across all experiments
    max_epochs = max(max(len(losses) for losses in experiment) for experiment in loss_lists)

    num_experiments = len(loss_lists)
    mean_losses = []
    std_losses = []

    for experiment_losses in loss_lists:
        # Create an array with shape (num_of_experiments, max_epochs), initialized with NaN
        losses_array = np.full((len(experiment_losses), max_epochs), np.nan)
        
        # Fill the array with loss values
        for i, losses in enumerate(experiment_losses):
            losses_array[i, :len(losses)] = losses
        
        # Compute mean and std deviation along the experiment axis, ignoring NaNs
        mean_losses.append(np.nanmean(losses_array, axis=0))
        std_losses.append(np.nanstd(losses_array, axis=0))
    
    # Plotting with seaborn
    plt.figure(figsize=(10, 6))
    epochs_x = np.arange(1, max_epochs + 1)
    colors = sns.color_palette("hsv", num_experiments)  # Using seaborn color palette
    
    for i, (mean_loss, std_loss) in enumerate(zip(mean_losses, std_losses)):
        plt.plot(epochs_x, mean_loss, label=f'Experiment Group {i+1}', color=colors[i])
        plt.fill_between(epochs_x, mean_loss - std_loss, mean_loss + std_loss, color=colors[i], alpha=0.3)
    
    plt.title('Mean Loss Per Epoch With Standard Deviation')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.xticks(epochs_x)  # Set x-ticks to show integer values for epochs
    plt.legend()
    plt.show()

def plot_loss(training_epoch_losses, validation_epoch_losses, plot_path):
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

def plot_lr_vs_loss(log_lrs, losses, plot_path):
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
    
    plt.plot(log_lrs, losses, label="Learning Rate vs Loss")
    
    plt.title("Learning Rate vs Loss")
    plt.xlabel("Log Learning Rate")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    # Deriving CSV path from the plot path
    csv_path = plot_path.replace(".png", ".csv")
    
    # Saving data to a CSV file
    data = {'Log Learning Rate': log_lrs, 'Loss': losses}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
