from trainers.base_trainer import BaseTrainer
from tqdm import tqdm
import torch
from torch_topological.nn import VietorisRipsComplex

class TopologicalTrainer(BaseTrainer):
    """
    A trainer class for training models with a topological regularization term.

    Args:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
        model_type (str, optional): The type of the model. Defaults to None.
    """

    def __init__(self, model, device, model_type=None):
        super().__init__(model, device)
        self.features = None
        self.model_type = model_type
        self._register_feature_hook()

    def _register_feature_hook(self):
        """
        Registers a forward hook to extract features from the model.
        """
        def hook(module, input, output):
            self.features = output

        if self.model_type == 'efficientnet':
            self.model._avg_pooling.register_forward_hook(hook)
        elif self.model_type == 'densenet':
            self.model.features.norm5.register_forward_hook(hook)

    def _topological_regularizer(self, features):
        """
        Computes the topological regularization loss based on the features.

        Args:
            features (torch.Tensor): The extracted features from the model.

        Returns:
            torch.Tensor: The topological regularization loss.
        """
        diagram_computator = VietorisRipsComplex(dim=1, keep_infinite_features=False, p=2)
        pd = diagram_computator(features)
        pd = torch.cat((pd[0].diagram, pd[1].diagram), 0)
        L = torch.max(pd[:, 1] - pd[:, 0])
        loss = torch.sum(pd[:, 1] - pd[:, 0]) / L
        return loss

    def _train_epoch(self, train_loader, epoch, num_epochs, verbose=True, **kwargs) -> float:
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): The data loader for training data.
            epoch (int): The current epoch number.
            num_epochs (int): The total number of epochs.
            verbose (bool, optional): Whether to display progress bar. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        alpha = kwargs.get('alpha', 0.0)
        
        if verbose:
            progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        else:
            progress_bar = enumerate(train_loader, 1)

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            topological_loss = self._topological_regularizer(outputs)
            total_loss = loss + alpha * topological_loss
            total_loss.backward()
            self.optimizer.step()
            running_loss += total_loss.item()

            if verbose:
                progress_bar.set_postfix({'loss': running_loss / batch_idx})

        epoch_loss = running_loss / len(train_loader)
    
        return epoch_loss
