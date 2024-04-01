import torchvision.models as models
import torch.nn as nn

def get_efficientnet(model_name, num_classes, pretrained=True):
    """
    Get an EfficientNet model with a custom classifier.

    Args:
        model_name (str): Name of the EfficientNet model to use. Supported options are "efficientnet_b0", "efficientnet_b1", and "efficientnet_b2".
        num_classes (int): Number of output classes for the custom classifier.
        pretrained (bool, optional): Whether to load pretrained weights for the model. Defaults to True.

    Returns:
        torch.nn.Module: EfficientNet model with a custom classifier.

    Raises:
        ValueError: If an unsupported EfficientNet version is specified.
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=weights)
    else:
        raise ValueError("Unsupported EfficientNet version")
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model
