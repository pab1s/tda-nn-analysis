import torch
from utils.plotting import plot_loss
from utils.logging import log_to_csv, log_epoch_results
from abc import ABC, abstractmethod
import time
from typing import Tuple


class BaseTrainer(ABC):
    """
    Base class for trainers in the tda-nn-separability project.

    Attributes:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
        criterion: The loss function used for training.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler.
        metrics (list): List of metrics used for evaluation during training.

    Methods:
        build: Build the model, criterion, optimizer, and scheduler.
        freeze_layers: Freeze layers up to a specified layer.
        unfreeze_all_layers: Unfreeze all layers of the model.
        train: Train the model for a given number of epochs.
        evaluate: Evaluate the model on a given dataset.
    """

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

    def train(self, train_loader, num_epochs, valid_loader=None, log_path=None, plot_path=None, verbose=True) -> None:
        """
        Train the model for a given number of epochs, calculating metrics at the end of each epoch
        for both training and validation sets.

        Args:
            train_loader: The data loader for the training set.
            num_epochs (int): The number of epochs to train the model.
            valid_loader: The data loader for the validation set (optional).
            log_path: The path to save the training log (optional).
            plot_path: The path to save the training plot (optional).
            verbose (bool): Whether to print training progress (default: True).
        """
        training_epoch_losses = []
        validation_epoch_losses = []
        metric_values = {metric.name: {'train': [], 'valid': []} for metric in self.metrics}

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_loss_train = self._train_epoch(train_loader, epoch, num_epochs, verbose)
            training_epoch_losses.append(epoch_loss_train)

            _, epoch_metrics_train = self.evaluate(train_loader, self.metrics, verbose=False)

            if valid_loader is not None:
                epoch_loss_valid, epoch_metrics_valid = self.evaluate(valid_loader, self.metrics, verbose=False)
                validation_epoch_losses.append(epoch_loss_valid)
            else:
                epoch_metrics_valid = {metric.name: None for metric in self.metrics}

            for metric_name in metric_values.keys():
                metric_values[metric_name]['train'].append(epoch_metrics_train[metric_name])
                metric_values[metric_name]['valid'].append(epoch_metrics_valid.get(metric_name))

            if verbose:
                log_epoch_results(epoch, num_epochs, epoch_loss_train, epoch_metrics_train, epoch_metrics_valid)

            if log_path is not None:
                log_to_csv(training_epoch_losses, validation_epoch_losses, metric_values, log_path)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Training completed in: {elapsed_time:.2f} seconds")

        if plot_path is not None:
            plot_loss(training_epoch_losses, validation_epoch_losses, plot_path)

    def evaluate(self, data_loader, metrics=None, verbose=True) -> Tuple[float, dict]:
        """
        Evaluate the model on a given dataset.

        Args:
            data_loader: The data loader for the dataset.
            metrics (list): List of metrics to evaluate (optional).
            verbose (bool): Whether to print evaluation results (default: True).

        Returns:
            Tuple[float, dict]: The average loss and metric results.
        """
        if metrics is None:
            metrics = self.metrics

        self.model.eval()
        total_loss = 0
        metric_results = {metric.name: 0 for metric in metrics}
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                for metric in metrics:
                    metric_value = metric(labels, outputs)
                    metric_results[metric.name] += metric_value

        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        for metric_name in metric_results.keys():
            metric_results[metric_name] /= num_batches

        if verbose:
            print(f"Evaluation - Loss: {avg_loss:.4f}")
            for metric_name, metric_value in metric_results.items():
                print(f"{metric_name}: {metric_value:.4f}")

        return avg_loss, metric_results
