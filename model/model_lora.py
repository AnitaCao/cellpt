import torch
import torch.nn as nn
from model.dinov2_model import DINOv2Classifier
from model.lora import LoRALinear

def replace_linear_with_lora(module, r=8, alpha=16):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            replace_linear_with_lora(child, r, alpha)  # recurse


def apply_lora_to_dinov2(model: DINOv2Classifier, r=8, alpha=16, target_layers=None):
    """
    Replace specific nn.Linear layers in DINOv2 encoder with LoRA-adapted versions.
    """

    if target_layers is None:
        # Default: all attention and MLP linear layers in all encoder blocks
        target_layers = []
        for i in range(12):
            target_layers += [
                f"backbone.model.encoder.layer.{i}.attention.attention.query",
                f"backbone.model.encoder.layer.{i}.attention.attention.value",
                f"backbone.model.encoder.layer.{i}.mlp.fc1",
                f"backbone.model.encoder.layer.{i}.mlp.fc2",
            ]

    for layer_name in target_layers:
        parts = layer_name.split(".")
        submodule = model
        for p in parts[:-1]:
            submodule = getattr(submodule, p)
        last_name = parts[-1]
        orig_layer = getattr(submodule, last_name)
        if isinstance(orig_layer, nn.Linear):
            setattr(submodule, last_name, LoRALinear(orig_layer, r=r, alpha=alpha))
        else:
            print(f"[WARN] Skipped {layer_name} — not nn.Linear")

    return model
