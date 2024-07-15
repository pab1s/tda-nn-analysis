from factories.factory import Factory
import torchvision.models as models
import torch.nn as nn
from typing import Any

class ModelFactory(Factory):
    """
    Factory class for creating different models.

    This class provides methods to register and create different models such as EfficientNet, ResNet, and DenseNet.
    Each model is registered with a name and a lambda function that takes the number of classes and a flag indicating
    whether to use pretrained weights. The lambda function then calls the corresponding `get_efficientnet`, `get_resnet`,
    or `get_densenet` method to create the model with the specified architecture and number of classes.

    Args:
        Factory (class): Base factory class.

    Attributes:
        None

    Methods:
        register_models: Register all the available models.
        get_efficientnet: Create an EfficientNet model with the specified architecture and number of classes.
        get_resnet: Create a ResNet model with the specified architecture and number of classes.
        get_densenet: Create a DenseNet model with the specified architecture and number of classes.
    """

    def __init__(self):
        super().__init__()
        self.register_models()

    def register_models(self) -> None:
        """
        Register all the available models.

        This method registers all the available models by calling the `register` method of the base factory class.
        Each model is registered with a name and a lambda function that takes the number of classes and a flag indicating
        whether to use pretrained weights. The lambda function then calls the corresponding `get_efficientnet`, `get_resnet`,
        or `get_densenet` method to create the model with the specified architecture and number of classes.

        Args:
            None

        Returns:
            None
        """
        # Register EfficientNet models
        self.register("efficientnet_b0", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b0", num_classes, pretrained))
        self.register("efficientnet_b1", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b1", num_classes, pretrained))
        self.register("efficientnet_b2", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b2", num_classes, pretrained))
        self.register("efficientnet_b3", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b3", num_classes, pretrained))
        self.register("efficientnet_b4", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b4", num_classes, pretrained))
        self.register("efficientnet_b5", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b5", num_classes, pretrained))
        self.register("efficientnet_b6", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b6", num_classes, pretrained))
        self.register("efficientnet_b7", lambda num_classes, pretrained=True: self.get_efficientnet("efficientnet_b7", num_classes, pretrained))
        
        # Register ResNet models
        self.register("resnet18", lambda num_classes, pretrained=True: self.get_resnet("resnet18", num_classes, pretrained))
        self.register("resnet34", lambda num_classes, pretrained=True: self.get_resnet("resnet34", num_classes, pretrained))
        self.register("resnet50", lambda num_classes, pretrained=True: self.get_resnet("resnet50", num_classes, pretrained))
        self.register("resnet101", lambda num_classes, pretrained=True: self.get_resnet("resnet101", num_classes, pretrained))
        self.register("resnet152", lambda num_classes, pretrained=True: self.get_resnet("resnet152", num_classes, pretrained))

        # Register DenseNet models
        self.register("densenet121", lambda num_classes, pretrained=True: self.get_densenet("densenet121", num_classes, pretrained))
        self.register("densenet161", lambda num_classes, pretrained=True: self.get_densenet("densenet161", num_classes, pretrained))
        self.register("densenet169", lambda num_classes, pretrained=True: self.get_densenet("densenet169", num_classes, pretrained))
        self.register("densenet201", lambda num_classes, pretrained=True: self.get_densenet("densenet201", num_classes, pretrained))

    def get_efficientnet(self, model_name, num_classes, pretrained) -> Any:
        """
        Create an EfficientNet model with the specified architecture and number of classes.

        This method creates an EfficientNet model with the specified architecture and number of classes.
        If the `pretrained` flag is set to True, the model is initialized with pretrained weights.
        The last fully connected layer of the model is replaced with a new fully connected layer that has
        the specified number of classes.

        Args:
            model_name (str): Name of the EfficientNet model architecture.
            num_classes (int): Number of classes for the classification task.
            pretrained (bool): Whether to use pretrained weights for the model.

        Returns:
            model (nn.Module): Created EfficientNet model.
        """
        model = models.__dict__[model_name](weights="DEFAULT" if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        return model

    def get_resnet(self, model_name, num_classes, pretrained) -> Any:
        """
        Create a ResNet model with the specified architecture and number of classes.

        This method creates a ResNet model with the specified architecture and number of classes.
        If the `pretrained` flag is set to True, the model is initialized with pretrained weights.
        The last fully connected layer of the model is replaced with a new fully connected layer that has
        the specified number of classes.

        Args:
            model_name (str): Name of the ResNet model architecture.
            num_classes (int): Number of classes for the classification task.
            pretrained (bool): Whether to use pretrained weights for the model.

        Returns:
            model (nn.Module): Created ResNet model.
        """
        model = models.__dict__[model_name](weights="DEFAULT" if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        return model

    def get_densenet(self, model_name, num_classes, pretrained) -> Any:
        """
        Create a DenseNet model with the specified architecture and number of classes.

        This method creates a DenseNet model with the specified architecture and number of classes.
        If the `pretrained` flag is set to True, the model is initialized with pretrained weights.
        The last fully connected layer of the model is replaced with a new fully connected layer that has
        the specified number of classes.

        Args:
            model_name (str): Name of the DenseNet model architecture.
            num_classes (int): Number of classes for the classification task.
            pretrained (bool): Whether to use pretrained weights for the model.

        Returns:
            model (nn.Module): Created DenseNet model.
        """
        model = models.__dict__[model_name](weights="DEFAULT" if pretrained else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        return model
    