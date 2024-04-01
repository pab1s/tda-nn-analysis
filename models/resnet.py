import torchvision.models as models
import torch.nn as nn

def get_resnet(model_name, num_classes, pretrained=True):
    """
    Get a ResNet model with a specified architecture.

    Args:
        model_name (str): The name of the ResNet architecture. Supported options are "resnet18", "resnet34", and "resnet50".
        num_classes (int): The number of output classes.
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.

    Returns:
        torch.nn.Module: The ResNet model with the specified architecture and number of output classes.

    Raises:
        ValueError: If an unsupported ResNet version is specified.
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("Unsupported ResNet version")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
