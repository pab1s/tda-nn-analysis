from factories.factory import Factory
from torch.optim import Adam, SGD

class OptimizerFactory(Factory):
    def __init__(self):
        super().__init__()
        self.register("Adam", Adam)
        self.register("SGD", SGD)

    def create(self, name, **kwargs):
        creator = self._creators.get(name)
        if not creator:
            raise ValueError(f"Unknown configuration: {name}")
        return creator
