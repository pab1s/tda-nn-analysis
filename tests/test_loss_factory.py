import pytest
from torch.nn import CrossEntropyLoss, MSELoss
from factories.loss_factory import LossFactory

def test_loss_factory_creation():
    """
    Test the creation of different loss functions using the LossFactory class.
    """
    
    factory = LossFactory()
    
    # Test CrossEntropyLoss creation
    loss = factory.create("CrossEntropyLoss")
    assert isinstance(loss, CrossEntropyLoss), "Failed to create CrossEntropyLoss"
    
    # Test MSELoss creation
    loss = factory.create("MSELoss")
    assert isinstance(loss, MSELoss), "Failed to create MSELoss"

def test_loss_factory_unknown():
    factory = LossFactory()
    with pytest.raises(ValueError) as e:
        factory.create("UnknownLoss")
    assert "Unknown configuration" in str(e.value)
