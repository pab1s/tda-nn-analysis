from models.efficientnet import get_efficientnet
from models.resnet import get_resnet

def get_model(model_name, num_classes, pretrained=True):
    """
    Returns a pre-trained model based on the given model_name.

    Args:
        model_name (str): The name of the model to be used.
        num_classes (int): The number of output classes.
        pretrained (bool, optional): Whether to load pre-trained weights. Defaults to True.

    Returns:
        torch.nn.Module: The pre-trained model.

    Raises:
        ValueError: If the model_name is not supported.
    """
    if 'efficientnet' in model_name:
        return get_efficientnet(model_name, num_classes, pretrained)
    elif 'resnet' in model_name:
        return get_resnet(model_name, num_classes, pretrained)
    else:
        raise ValueError("Model not supported")