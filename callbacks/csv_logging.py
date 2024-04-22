from callbacks.callback import Callback
import csv

class CSVLogging(Callback):
    """
    Callback for logging training and validation metrics to a CSV file.

    Args:
        csv_path (str): The path to the CSV file.

    Attributes:
        csv_path (str): The path to the CSV file.
        headers_written (bool): Flag indicating whether the headers have been written to the CSV file.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.headers_written = False

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called at the end of each epoch during training.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing the training and validation metrics.

        Returns:
            None
        """
        if logs is None:
            return
        
        epoch_data = logs.get('epoch')
        train_loss = logs.get('train_loss')
        val_loss = logs.get('val_loss')
        train_metrics = logs.get('train_metrics', {})
        val_metrics = logs.get('val_metrics', {})

        metrics = {'train_loss': train_loss, 'val_loss': val_loss}
        metrics.update({f'train_{key}': value for key, value in train_metrics.items()})
        metrics.update({f'val_{key}': value for key, value in val_metrics.items()})

        if not self.headers_written:
            headers = ['epoch'] + list(metrics.keys())
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                self.headers_written = True

        values = [epoch_data] + [metrics[key] for key in headers[1:]]  # Ensure the order matches headers
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(values)
