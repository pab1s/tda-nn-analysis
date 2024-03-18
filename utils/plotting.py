import matplotlib.pyplot as plt
import numpy as np

def plot_loss(epoch_losses, plot_path):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(epoch_losses) + 1), epoch_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
