import torch
from utils.plotting import plot_loss
from utils.logging import log_to_csv

class BaseTrainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = None
        self.optimizer = None

    def build(self, criterion, optimizer, scheduler=None):
        """ Set the loss function, optimizer, and learning rate scheduler. """

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _train_epoch(self, train_loader, epoch, num_epochs, verbose=True):
        """ Train the model for one epoch. """

        raise NotImplementedError("The train_epoch method must be implemented by the subclass.")

    def train(self, train_loader, num_epochs, log_path, plot_path, verbose=True):
        """ Train the model for a given number of epochs. """

        epoch_losses = []

        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(train_loader, epoch, num_epochs, verbose)
            epoch_losses.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            log_to_csv(epoch_losses, log_path)

        if verbose:
            plot_loss(epoch_losses, plot_path)

    def evaluate(self, test_loader, verbose=True):
        """ Evaluate the model on the test set. """

        correct = 0
        total = 0
        
        self.model.eval()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        if verbose:
            print(f"Accuracy: {accuracy}%")
        return accuracy
