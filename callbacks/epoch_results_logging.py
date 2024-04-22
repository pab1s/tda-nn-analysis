from callbacks.callback import Callback

class EpochResultsLogging(Callback):
    """
    Callback for logging epoch results during training.

    This callback prints the training loss, training metrics, and validation metrics at the end of each epoch.

    Args:
        Callback: The base class for Keras callbacks.

    Methods:
        on_epoch_end: Called at the end of each epoch to print the epoch results.
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Prints the epoch results at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): A dictionary containing the training and validation metrics.

        Returns:
            None
        """
        epoch_loss_train = logs.get('train_loss')
        epoch_metrics_train = logs.get('train_metrics', {})
        epoch_metrics_valid = logs.get('val_metrics', {})
        num_epochs = logs.get('num_epochs', 0)

        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {epoch_loss_train:.4f}")

        for metric_name, value in epoch_metrics_train.items():
            print(f"Training {metric_name}: {value:.4f}")

        if epoch_metrics_valid:
            for metric_name, value in epoch_metrics_valid.items():
                print(f"Validation {metric_name}: {value if value is not None else 'N/A'}")
