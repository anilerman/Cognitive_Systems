# model.py

import torchvision.models as models
import torch.nn as nn

def get_model(config, num_classes):
    model_name = config.get("name", "resnet18")  # Default to resnet18 if no name specified
    pretrained = config.get("pretrained", True)

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError(f"{model_name} is not implemented or supported.")

    # Replace the final fully connected layer for transfer learning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
