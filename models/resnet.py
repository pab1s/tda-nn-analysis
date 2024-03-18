import torchvision.models as models
import torch.nn as nn

def get_resnet(model_name, num_classes, pretrained=True):
    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet34":
        model = models.resnet34(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("Unsupported ResNet version")
    
    # Change the classifier head to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
