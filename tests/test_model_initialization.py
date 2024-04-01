import pytest
import torch
from models.efficientnet import get_efficientnet
from models.resnet import get_resnet

@pytest.mark.parametrize("model_func, model_name", [
    (get_efficientnet, 'efficientnet_b0'),
    (get_resnet, 'resnet18'),
])
def test_model_initialization_and_forward_pass(model_func, model_name):
    model = model_func(model_name, num_classes=10, pretrained=False)
    assert model is not None, f"{model_name} should be initialized"
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 10), f"Output shape of {model_name} should be (2, 10) for batch size of 2 and 10 classes"
