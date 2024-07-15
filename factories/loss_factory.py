from factories.factory import Factory
from torch.nn import CrossEntropyLoss, MSELoss

class LossFactory(Factory):
    """
    Factory class for creating different loss functions.
    """

    def __init__(self):
        super().__init__()
        self.register("CrossEntropyLoss", lambda **kwargs: CrossEntropyLoss(**kwargs))
        self.register("MSELoss", lambda **kwargs: MSELoss(**kwargs))