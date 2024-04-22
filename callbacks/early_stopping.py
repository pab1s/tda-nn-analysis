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

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        score = -current if 'loss' in self.monitor else current
        if self.best_score is None:
            self.best_score = score
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

    def should_continue(self, logs=None):
        return not self.early_stop
