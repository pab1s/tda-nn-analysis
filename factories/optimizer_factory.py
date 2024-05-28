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
    
    def update(self, optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
