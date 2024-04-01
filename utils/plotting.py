import matplotlib.pyplot as plt
import numpy as np

def plot_loss(training_epoch_losses, validation_epoch_losses, plot_path):
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
