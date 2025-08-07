import torch
import os

def save_full_model(model, path):
    """
    Save full model including backbone, LoRA adapters, and classifier head.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[✔] Full model saved to {path}")


def load_full_model(model, path, map_location="cpu"):
    """
    Load full model including backbone + LoRA adapters.
    Must match architecture exactly.
    """
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"[✔] Model loaded from {path}")
    return model


def save_lora_only(model, path):
    """
    Save only LoRA adapter weights from the model.
    This is a subset of the full model state dict.
    """
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_weights[name] = param.detach().cpu()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(lora_weights, path)
    print(f"[✔] LoRA-only weights saved to {path}")


def load_lora_only(model, path, strict=True):
    """
    Load only LoRA adapter weights into an existing model.
    """
    lora_weights = torch.load(path)
    missing = model.load_state_dict(lora_weights, strict=False)
    print(f"[✔] LoRA-only weights loaded from {path}")
    if strict and (missing.missing_keys or missing.unexpected_keys):
        print("[!] Warning: Missing or unexpected keys:")
        print("Missing:", missing.missing_keys)
        print("Unexpected:", missing.unexpected_keys)
    return model
