import pytest
from torchvision.models import efficientnet_b0, resnet18
from factories.model_factory import ModelFactory

def test_model_factory_creation():
    """
    Test the creation of models using the ModelFactory class.
    """
    
    factory = ModelFactory()
    
    # Test EfficientNet B0 creation
    model = factory.create("efficientnet_b0", num_classes=10, pretrained=False)
    assert isinstance(model, efficientnet_b0().__class__), "Failed to create EfficientNet B0 with correct class"
    
    # Test ResNet18 creation
    model = factory.create("resnet18", num_classes=10, pretrained=False)
    assert isinstance(model, resnet18().__class__), "Failed to create ResNet18 with correct class"

def test_model_factory_unknown_model():
    factory = ModelFactory()
    with pytest.raises(ValueError) as e:
        factory.create("unknown_model", num_classes=10, pretrained=True)
    assert "Unknown configuration" in str(e.value)
