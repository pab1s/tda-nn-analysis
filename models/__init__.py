from models.efficientnet import get_efficientnet
from models.resnet import get_resnet

def get_model(model_name, num_classes, pretrained=True):
    if 'efficientnet' in model_name:
        return get_efficientnet(model_name, num_classes, pretrained)
    elif 'resnet' in model_name:
        return get_resnet(model_name, num_classes, pretrained)
    else:
        raise ValueError("Model not supported")