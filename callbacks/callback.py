class Callback:
    def should_continue(self) -> bool:
        return True
    
    def on_epoch_begin(self, epoch, logs=None) -> None:
        pass

    def on_epoch_end(self, epoch, logs=None) -> None:
        pass

    def on_train_begin(self, logs=None) -> None:
        pass

    def on_train_end(self, logs=None) -> None:
        pass
