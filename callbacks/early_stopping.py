import torch
from copy import deepcopy
from callbacks.callback import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=5, verbose=False, delta=0):
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.stopped_epoch = 0
        self.early_stop = False
        self.best_model_state = None
        self.best_optimizer_state = None
        self.model = None
        self.optimizer = None

    def set_model_and_optimizer(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def on_epoch_end(self, epoch, logs=None):
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be set before calling on_epoch_end.")

        current = logs.get(self.monitor)
        if current is None:
            return

        score = -current if 'loss' in self.monitor else current
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
        elif score < self.best_score + self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch + 1
                if self.verbose:
                    print(f"Early stopping triggered at epoch {self.stopped_epoch}")
        else:
            self.best_score = score
            self.epochs_no_improve = 0
            self.save_checkpoint()

    def save_checkpoint(self):
        self.best_model_state = deepcopy(self.model.state_dict())
        self.best_optimizer_state = deepcopy(self.optimizer.state_dict())

    def should_continue(self, logs=None):
        return not self.early_stop

    def load_best_checkpoint(self):
        if self.best_model_state is None or self.best_optimizer_state is None:
            raise ValueError("No best checkpoint available to load.")
        
        self.model.load_state_dict(self.best_model_state)
        self.optimizer.load_state_dict(self.best_optimizer_state)
        if self.verbose:
            print(f"Loaded best checkpoint")

    def on_train_end(self, logs=None):
        self.load_best_checkpoint()
        if self.verbose:
            print(f"Training ended. Best model checkpoint has been loaded.")
