from callbacks.callback import Callback
import os
import torch

class Checkpoint(Callback):
    """
    Callback class for saving model checkpoints during training.

    Args:
        checkpoint_dir (str): Directory to save the checkpoints.
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to be saved. Default is None.
        save_freq (int, optional): Frequency of saving checkpoints. Default is 1.
        verbose (bool, optional): Whether to print the checkpoint save path. Default is False.
    """

    def __init__(self, checkpoint_dir, model, optimizer, scheduler=None, save_freq=5, verbose=False):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_freq = save_freq
        self.verbose = verbose
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback function called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary containing training and validation losses. Default is None.
        """
        if (epoch + 1) % self.save_freq == 0:
            self.save_checkpoint(epoch, logs)

    def save_checkpoint(self, epoch, logs=None):
        """
        Save the model checkpoint.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary containing training and validation losses. Default is None.
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': logs.get('training_losses', []),
            'val_losses': logs.get('validation_losses', []),
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(state, save_path)
        if self.verbose:
            print(f"Checkpoint saved at {save_path}")
