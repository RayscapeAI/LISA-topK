import torch
import torchvision
import torch.nn as nn
import timm

from utils.config import device, classes
from utils.model_style import DenseNet121_IBN_proposed


def get_densenet121(checkpoint_path=None):
    model = torchvision.models.densenet121(
        drop_rate=0.3, weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    )

    if checkpoint_path is not None:
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=17, bias=True
        )
        state_dict = torch.load(checkpoint_path)["model"]
        model.load_state_dict(state_dict)

    model.classifier = nn.Linear(
        in_features=model.classifier.in_features, out_features=len(classes), bias=True
    )

    model = model.to(device)
    return model


def get_style_densenet121(checkpoint_path=None):
    model = DenseNet121_IBN_proposed(num_classes=len(classes))
    model = model.to(device)
    return model
