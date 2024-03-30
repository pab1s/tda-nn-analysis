import torch
from utils.plotting import plot_loss
from utils.logging import log_to_csv
import time


class BaseTrainer():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def build(self, criterion, optimizer_class, optimizer_params={}, scheduler=None, freeze_until_layer=None) -> None:
        """ Build the model, criterion, optimizer and scheduler. """
        self.criterion = criterion
        self.scheduler = scheduler

        if freeze_until_layer is not None:
            self.freeze_layers(freeze_until_layer=freeze_until_layer)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizer_class(trainable_params, **optimizer_params)

    def freeze_layers(self, freeze_until_layer=None) -> None:
        """Freeze layers up to a specified layer."""
        for name, param in self.model.named_parameters():
            if freeze_until_layer is None or name == freeze_until_layer:
                break
            param.requires_grad = False

    def unfreeze_all_layers(self) -> None:
        """Unfreeze all layers of the model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def _train_epoch(self, train_loader, epoch, num_epochs, verbose=True) -> float:
        """ Train the model for one epoch. """
        raise NotImplementedError(
            "The train_epoch method must be implemented by the subclass.")

    def train(self, train_loader, num_epochs, log_path=None, plot_path=None, verbose=True) -> None:
        """ Train the model for a given number of epochs. """
        training_epoch_losses = []
        valid_epoch_losses = []
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(
                train_loader, epoch, num_epochs, verbose)
            training_epoch_losses.append(epoch_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Training loss: {epoch_loss:.4f}")
                
            if log_path is not None:
                log_to_csv(training_epoch_losses, log_path)

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"Training completed in: {elapsed_time:.2f} seconds")

        if plot_path is not None:
            plot_loss(training_epoch_losses, plot_path)

    def evaluate(self, test_loader, verbose=True) -> float:
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
