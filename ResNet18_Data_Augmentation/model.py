import torch
import torchvision.models as models
import torch.nn as nn

def get_model(config, num_classes):
    model_name = config["name"]
    pretrained = config.get("pretrained", False)

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model