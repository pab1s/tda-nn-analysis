import torchvision.models as models
import torch.nn as nn

def get_efficientnet(model_name, num_classes, pretrained=True):
    weights = "DEFAULT" if pretrained else None

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=weights)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=weights)
    else:
        raise ValueError("Unsupported EfficientNet version")
    
    # Change the classifier head to match the number of classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model
