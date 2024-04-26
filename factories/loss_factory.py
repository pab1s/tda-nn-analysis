from factories.factory import Factory
from torch.nn import CrossEntropyLoss, MSELoss

class LossFactory(Factory):
    def __init__(self):
        super().__init__()
        self.register("CrossEntropyLoss", lambda: CrossEntropyLoss())
        self.register("MSELoss", lambda: MSELoss())
