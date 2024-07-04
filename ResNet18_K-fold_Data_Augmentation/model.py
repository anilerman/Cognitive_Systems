import torchvision.models as models
import torch.nn as nn

def get_model(config, num_classes):
    model_name = config.get("name", "resnet18")
    pretrained = config.get("pretrained", True)

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError(f"{model_name} is not implemented or supported.")


    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)

    return model
