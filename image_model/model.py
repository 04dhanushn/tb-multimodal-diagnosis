import torch.nn as nn
from torchvision import models

def get_image_model(num_classes):
    model = models.efficientnet_v2_s(weights="DEFAULT")
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
