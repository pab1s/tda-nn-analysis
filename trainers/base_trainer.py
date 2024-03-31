import torch
from utils.plotting import plot_loss
from utils.logging import log_to_csv
from abc import ABC, abstractmethod
import time


class BaseTrainer(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = []

    def build(self, criterion, optimizer_class, optimizer_params={}, scheduler=None, freeze_until_layer=None, metrics=[]) -> None:
        """ Build the model, criterion, optimizer and scheduler. """
        self.criterion = criterion
        self.scheduler = scheduler
        self.metrics = metrics

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

    @abstractmethod
    def _train_epoch(self, train_loader, epoch, num_epochs, verbose=True) -> float:
        """ Train the model for one epoch. """
        raise NotImplementedError(
            "The train_epoch method must be implemented by the subclass.")

    def train(self, train_loader, num_epochs, log_path=None, plot_path=None, verbose=True) -> None:
        """Train the model for a given number of epochs, calculating metrics at the end of each epoch."""
        training_epoch_losses = []
        metric_values = {metric.name: [] for metric in self.metrics}

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(train_loader, epoch, num_epochs, verbose)
            training_epoch_losses.append(epoch_loss)

            # Reset metrics at the start of each epoch
            epoch_metric_values = {metric.name: 0 for metric in self.metrics}
            self.model.eval()

            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)

                    for metric in self.metrics:
                        metric_value = metric(labels, outputs)
                        epoch_metric_values[metric.name] += metric_value

            # Average metric values across batches and log them
            num_batches = len(train_loader)
            for metric_name in epoch_metric_values.keys():
                epoch_metric_values[metric_name] /= num_batches
                metric_values[metric_name].append(epoch_metric_values[metric_name])

                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}, {metric_name}: {epoch_metric_values[metric_name]:.4f}")

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {epoch_loss:.4f}")

            self.model.train()

            if log_path is not None:
                log_to_csv(training_epoch_losses, log_path)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Training completed in: {elapsed_time:.2f} seconds")

        if plot_path is not None:
            plot_loss(training_epoch_losses, plot_path)

    def evaluate(self, test_loader, metrics=[], verbose=True) -> dict:
        """ Evaluate the model on the test set using provided metrics. """
        if len(metrics) > 0:
            self.metrics = metrics

        self.model.eval()
        metrics_results = {metric.name: 0 for metric in self.metrics}

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                for metric in self.metrics:
                    metric_value = metric(labels, outputs)
                    metrics_results[metric.name] += metric_value

        num_batches = len(test_loader)
        for metric in self.metrics:
            metrics_results[metric.name] /= num_batches

        if verbose:
            for metric_name, metric_value in metrics_results.items():
                print(f"{metric_name}: {metric_value:.4f}")

        return metrics_results
