from factories.factory import Factory
import torchvision.models as models
import torch.nn as nn

class ModelFactory(Factory):
    def __init__(self):
        super().__init__()
        self.register_models()

    def register_models(self):
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

    def get_efficientnet(self, model_name, num_classes, pretrained):
        model = models.__dict__[model_name](weights="DEFAULT" if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model

    def get_resnet(self, model_name, num_classes, pretrained):
        model = models.__dict__[model_name](weights="DEFAULT" if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
