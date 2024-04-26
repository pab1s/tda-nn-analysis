from factories.factory import Factory
from callbacks import CSVLogging, EarlyStopping

class CallbackFactory(Factory):
    def __init__(self):
        super().__init__()
        self.register("CSVLogging", CSVLogging)
        self.register("EarlyStopping", EarlyStopping)

    def create(self, name, **kwargs):
        creator = self._creators.get(name)
        if not creator:
            raise ValueError(f"Unknown callback: {name}")
        return creator(**kwargs)
