import csv


import csv

def log_to_csv(training_losses, validation_losses, metric_values, csv_path) -> None:
    """
    Logs the training and validation losses, along with metric values, to a CSV file.

    Args:
        training_losses (list): A list of training losses for each epoch.
        validation_losses (list): A list of validation losses for each epoch.
        metric_values (dict): A dictionary containing metric values for each epoch.
            The keys are the names of the metrics, and the values are dictionaries
            with 'train' and 'valid' keys, representing the metric values for training
            and validation sets, respectively.
        csv_path (str): The path to the CSV file where the log will be saved.
    """
    headers = ['epoch', 'train_loss', 'val_loss']
    metric_names = list(metric_values.keys())
    for name in metric_names:
        headers.append(f'train_{name}')
        headers.append(f'val_{name}')

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        num_epochs = max(len(training_losses), len(validation_losses))

        for epoch in range(num_epochs):
            row = [epoch + 1]
            row.append(training_losses[epoch] if epoch < len(training_losses) else 'N/A')
            row.append(validation_losses[epoch] if epoch < len(validation_losses) else 'N/A')

            for name in metric_names:
                train_metrics = metric_values[name]['train']
                valid_metrics = metric_values[name]['valid']

                row.append(train_metrics[epoch] if epoch < len(train_metrics) else 'N/A')
                row.append(valid_metrics[epoch] if epoch < len(valid_metrics) else 'N/A')
        
            writer.writerow(row)

def log_epoch_results(epoch, num_epochs, epoch_loss_train, epoch_metrics_train, epoch_metrics_valid=None) -> None:
    """
    Logs the results of an epoch during training.

    Args:
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.
        epoch_loss_train (float): The training loss for the epoch.
        epoch_metrics_train (dict): A dictionary containing the training metrics for the epoch.
        epoch_metrics_valid (dict, optional): A dictionary containing the validation metrics for the epoch. Defaults to None.
    """
    print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {epoch_loss_train:.4f}")
    
    for metric_name, value in epoch_metrics_train.items():
        print(f"Training {metric_name}: {value:.4f}")

    if epoch_metrics_valid:
        for metric_name, value in epoch_metrics_valid.items():
            print(f"Validation {metric_name}: {value if value is not None else 'N/A'}")
