from factories.factory import Factory
from callbacks import CSVLogging, EarlyStopping, Checkpoint

class CallbackFactory(Factory):
    """
    A factory class for creating callbacks.

    This class provides a way to register and create different types of callbacks.

    Attributes:
        _creators (dict): A dictionary mapping callback names to their corresponding creator functions.
    """

    def __init__(self):
        super().__init__()
        self.register("CSVLogging", CSVLogging)
        self.register("EarlyStopping", EarlyStopping)
        self.register("Checkpoint", Checkpoint)

    def create(self, name, **kwargs):
        """
        Create a callback instance.

        Args:
            name (str): The name of the callback to create.
            **kwargs: Additional keyword arguments to pass to the callback creator.

        Returns:
            object: An instance of the specified callback.

        Raises:
            ValueError: If the specified callback name is unknown.
        """
        creator = self._creators.get(name)
        if not creator:
            raise ValueError(f"Unknown callback: {name}")
        return creator(**kwargs)
