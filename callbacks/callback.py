class Callback:
    """
    A base class for defining callbacks in a training process.

    Callbacks are functions that can be executed at various stages during training.
    They can be used to perform additional actions or modify the behavior of the training process.

    Methods:
        should_continue(logs=None) -> bool:
            Determines whether the training process should continue or stop.

        on_epoch_begin(epoch, logs=None) -> None:
            Executed at the beginning of each epoch.

        on_epoch_end(epoch, logs=None) -> None:
            Executed at the end of each epoch.

        on_train_begin(logs=None) -> None:
            Executed at the beginning of the training process.

        on_train_end(logs=None) -> None:
            Executed at the end of the training process.
    """

    def should_continue(self, logs=None) -> bool:
        """
        Determines whether the training process should continue or stop.

        Args:
            logs (dict): Optional dictionary containing training logs.

        Returns:
            bool: True if the training process should continue, False otherwise.
        """
        return True
    
    def on_epoch_begin(self, epoch, logs=None) -> None:
        """
        Executed at the beginning of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Optional dictionary containing training logs.
        """
        pass

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        Executed at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict): Optional dictionary containing training logs.
        """
        pass

    def on_train_begin(self, logs=None) -> None:
        """
        Executed at the beginning of the training process.

        Args:
            logs (dict): Optional dictionary containing training logs.
        """
        pass

    def on_train_end(self, logs=None) -> None:
        """
        Executed at the end of the training process.

        Args:
            logs (dict): Optional dictionary containing training logs.
        """
        pass
