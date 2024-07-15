from factories.factory import Factory
from torch.optim import Adam, SGD
from typing import Any

class OptimizerFactory(Factory):
    """
    A factory class for creating optimizer instances.

    This factory allows registering and creating different optimizer instances,
    such as Adam and SGD.

    Attributes:
        _creators (dict): A dictionary mapping optimizer names to their corresponding creator functions.
    """

    def __init__(self):
        super().__init__()
        self.register("Adam", Adam)
        self.register("SGD", SGD)

    def create(self, name, **kwargs) -> Any:
        """
        Create an optimizer instance.

        Args:
            name (str): The name of the optimizer to create.
            **kwargs: Additional keyword arguments to be passed to the optimizer constructor.

        Returns:
            An instance of the specified optimizer.

        Raises:
            ValueError: If the specified optimizer name is not registered.
        """
        creator = self._creators.get(name)
        if not creator:
            raise ValueError(f"Unknown configuration: {name}")
        return creator
    
    def update(self, optimizer, new_lr) -> None:
        """
        Update the learning rate of an optimizer.

        Args:
            optimizer: The optimizer instance to update.
            new_lr (float): The new learning rate value.

        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
