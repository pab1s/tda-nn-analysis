from trainers.base_trainer import BaseTrainer
from tqdm import tqdm

class BasicTrainer(BaseTrainer):
    def __init__(self, model, device):
        super().__init__(model, device)

    def _train_epoch(self, train_loader, epoch, num_epochs, verbose=True):
        self.model.train()
        running_loss = 0.0
        if verbose:
            progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
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
    
