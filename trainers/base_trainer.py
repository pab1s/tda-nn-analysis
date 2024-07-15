import torch
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
        load_checkpoint: Loads a checkpoint and resumes training or evaluation.
        _train_epoch: Trains the model for one epoch using the provided train_loader.
        predict: Predict the output of the model for a given instance.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = []

    def load_checkpoint(self, load_path) -> dict:
        """
        Loads a checkpoint and resumes training or evaluation.

        Args:
            load_path (str): Path to the checkpoint file.

        Returns:
            dict: The loaded checkpoint state.
        """
        state = torch.load(load_path)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        return state

    def build(self, criterion, optimizer_class, optimizer_params={}, scheduler=None, freeze_until_layer=None, metrics=[]) -> None:
        """
        Build the model, criterion, optimizer, and scheduler.

        Args:
            criterion: The loss function used for training.
            optimizer_class: The optimizer class to be used.
            optimizer_params (dict): Additional parameters for the optimizer (default: {}).
            scheduler: The learning rate scheduler (default: None).
            freeze_until_layer: The layer until which to freeze the model (default: None).
            metrics (list): List of metrics used for evaluation during training (default: []).

        Returns:
            None
        """
        self.criterion = criterion
        self.scheduler = scheduler
        self.metrics = metrics

        if freeze_until_layer is not None:
            self.freeze_layers(freeze_until_layer=freeze_until_layer)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optimizer_class(trainable_params, **optimizer_params)

    def freeze_layers(self, freeze_until_layer=None) -> None:
        """
        Freeze layers up to a specified layer.

        Args:
            freeze_until_layer: The layer until which to freeze the model (default: None).

        Returns:
            None
        """
        for name, param in self.model.named_parameters():
            print(name)
            if freeze_until_layer is None or name == freeze_until_layer:
                break
            param.requires_grad = False

    def unfreeze_all_layers(self) -> None:
        """
        Unfreeze all layers of the model.

        Returns:
            None
        """
        for param in self.model.parameters():
            param.requires_grad = True

    @abstractmethod
    def _train_epoch(self, train_loader, epoch, num_epochs, **kwargs) -> float:
        """
        Trains the model for one epoch using the provided train_loader.

        Args:
            train_loader (DataLoader): The data loader for training data.
            epoch (int): The current epoch number.
            num_epochs (int): The total number of epochs.

        Returns:
            float: The loss value for the epoch.
        """
        raise NotImplementedError("The train_epoch method must be implemented by the subclass.")

    def train(self, train_loader, num_epochs, valid_loader=None, callbacks=None, **kwargs) -> None:
        """
        Train the model for a given number of epochs, calculating metrics at the end of each epoch
        for both training and validation sets.

        Args:
            train_loader: The data loader for the training set.
            num_epochs (int): The number of epochs to train the model.
            valid_loader: The data loader for the validation set (optional).
            callbacks: List of callback objects to use during training (optional).

        Returns:
            None
        """
        logs = {}
        times = []
        training_epoch_losses = []
        validation_epoch_losses = []

        if callbacks is None:
            callbacks = []

        start_time = time.time()

        for callback in callbacks:
            callback.on_train_begin(logs=logs)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs=logs)

            logs['epoch'] = epoch
            epoch_loss_train = self._train_epoch(train_loader, epoch, num_epochs, **kwargs)
            training_epoch_losses.append(epoch_loss_train)

            _, epoch_metrics_train = self.evaluate(train_loader, self.metrics, verbose=False)
            logs['train_loss'] = epoch_loss_train
            logs['train_metrics'] = epoch_metrics_train

            if valid_loader is not None:
                epoch_loss_valid, epoch_metrics_valid = self.evaluate(valid_loader, self.metrics, verbose=False)
                validation_epoch_losses.append(epoch_loss_valid)
                logs['val_loss'] = epoch_loss_valid
                logs['val_metrics'] = epoch_metrics_valid
            else:
                logs['val_loss'] = None
                logs['val_metrics'] = {}

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)

            epoch_time = time.time() - epoch_start_time
            times.append(epoch_time)

            if not all(callback.should_continue(logs=logs) for callback in callbacks):
                print(f"Training stopped early at epoch {epoch + 1}.")
                break

        logs['times'] = times

        for callback in callbacks:
            callback.on_train_end(logs=logs)

        elapsed_time = time.time() - start_time
        print(f"Training completed in: {elapsed_time:.2f} seconds")

    def predict(self, instance) -> torch.Tensor:
        """
        Predict the output of the model for a given instance.

        Args:
            instance: The input instance to predict.

        Returns:
            torch.Tensor: The model's prediction.
        """
        self.model.eval()
        with torch.no_grad():
            instance = instance.to(self.device)
            output = self.model(instance)
        return output

    def evaluate(self, data_loader, metrics=None, verbose=True) -> Tuple[float, dict]:
        """
        Evaluate the model on a given dataset.

        Args:
            data_loader: The data loader for the dataset.
            metrics (list): List of metrics used for evaluation (default: None).
            verbose (bool): Whether to print evaluation results (default: True).

        Returns:
            Tuple[float, dict]: The average loss and metric results.
        """
        if metrics is None:
            metrics = self.metrics

        self.model.eval()
        total_loss = 0
        metric_aggregators = {metric.name: [] for metric in metrics}

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                batch_size = labels.size(0)

                for metric in metrics:
                    metric_value = metric(labels, outputs)
                    metric_aggregators[metric.name].append((metric_value, batch_size))

        num_batches = len(data_loader)
        avg_loss = total_loss / num_batches
        metric_results = {}

        for metric_name, values in metric_aggregators.items():
            if any(v[1] for v in values):
                weighted_sum = sum(v[0] * v[1] for v in values)
                total_weight = sum(v[1] for v in values)
                metric_results[metric_name] = weighted_sum / total_weight
            else:
                metric_results[metric_name] = 0

        if verbose:
            print(f"Evaluation - Loss: {avg_loss:.4f}")
            for metric_name, metric_value in metric_results.items():
                print(f"{metric_name}: {metric_value:.4f}")

        return avg_loss, metric_results