import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model


class DINOv2Backbone(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze=True):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0]  # CLS token


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes, model_name="facebook/dinov2-base", freeze=True):
        super().__init__()
        self.backbone = DINOv2Backbone(model_name=model_name, freeze=freeze)
        self.classifier = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def get_model(num_classes, freeze_backbone=True):
    return DINOv2Classifier(num_classes=num_classes, freeze=freeze_backbone)
