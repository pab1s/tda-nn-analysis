from trainers.base_trainer import BaseTrainer
from tqdm import tqdm


class BasicTrainer(BaseTrainer):
    """
    A basic trainer class for training a model.

    Args:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.

    Attributes:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
    """

    def __init__(self, model, device):
        super().__init__(model, device)

    def _train_epoch(self, train_loader, epoch, num_epochs, verbose=True) -> float:
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): The data loader for training data.
            epoch (int): The current epoch number.
            num_epochs (int): The total number of epochs.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        if verbose:
            progress_bar = tqdm(enumerate(train_loader, 1), total=len(
                train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        else:
            progress_bar = enumerate(train_loader, 1)

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if verbose:
                progress_bar.set_postfix({'loss': running_loss / batch_idx})

        epoch_loss = running_loss / len(train_loader)
    
        return epoch_loss
