import torch
import torch.nn as nn
import torchvision

class FusionModel(nn.Module):
    def __init__(self, tabular_model, image_weights):
        super().__init__()

        self.image_model = torchvision.models.efficientnet_v2_s(weights=None)
        self.image_model.classifier = nn.Identity()
        self.image_model.load_state_dict(image_weights, strict=False)

        self.img_fc = nn.Linear(1280, 128)
        self.tab_model = tabular_model

        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, tab, img):
        img_feat = self.img_fc(self.image_model(img))
        tab_feat = self.tab_model(tab)
        return self.classifier(torch.cat([img_feat, tab_feat], dim=1)).squeeze(1)
