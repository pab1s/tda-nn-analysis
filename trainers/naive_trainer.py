from trainers.base_trainer import BaseTrainer


class NaiveTrainer(BaseTrainer):
    """
    A complete naive trainer class for training a model. It main purpose is to
    be used as a boilerplate to test some functionalities of the trainer.

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
        Simulates training the model for one epoch. It actually does nothing.

        Args:
            train_loader (DataLoader): The data loader for training data.
            epoch (int): The current epoch number.
            num_epochs (int): The total number of epochs.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.

        Returns:
            float: A mock loss value for the epoch of 0.0.
        """
        return 0.0