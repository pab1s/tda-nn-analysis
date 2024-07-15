import pytest
import torch
from torch.optim import Adam, SGD
from factories.optimizer_factory import OptimizerFactory

def test_optimizer_factory_creation():
    """
    Test the creation of optimizers using the OptimizerFactory class.
    """
    
    factory = OptimizerFactory()

    # Simulate model parameters
    params = [torch.tensor([1.0, 2.0], requires_grad=True)]
    
    # Test Adam creation with a learning rate
    optimizer = factory.create("Adam")
    assert isinstance(optimizer(lr=0.01, params=params), Adam), "Failed to create Adam optimizer"
    
    # Test SGD creation with a learning rate
    optimizer = factory.create("SGD")
    assert isinstance(optimizer(lr=0.01, params=params), SGD), "Failed to create SGD optimizer"

def test_optimizer_factory_unknown():
    factory = OptimizerFactory()
    with pytest.raises(ValueError) as excinfo:
        factory.create("UnknownOptimizer")
    assert "Unknown configuration" in str(excinfo.value)
