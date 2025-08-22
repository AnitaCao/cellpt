# model/lora.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=8, alpha=16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(original_linear, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear")

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # freeze original weights
        self.weight = original_linear.weight
        self.bias   = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)

        # LoRA adapters
        self.lora_down = nn.Linear(self.in_features, self.r, bias=False)
        self.lora_up   = nn.Linear(self.r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = self.lora_up(self.dropout(self.lora_down(x)))
        return base + self.scaling * lora

def apply_lora_to_timm_vit(model, last_n_blocks=6, r=8, alpha=16, dropout=0.0):
    """
    Add LoRA to qkv/proj and MLP fc1/fc2 of the last N blocks of a timm ViT/DINOv2.
    Returns a list of LoRA parameters for the optimizer.
    """
    if not hasattr(model, "blocks"):
        raise RuntimeError("apply_lora_to_timm_vit: model has no 'blocks'; pass the ViT itself.")

    total = len(model.blocks)
    start = max(0, total - last_n_blocks)
    lora_params = []

    for i in range(start, total):
        blk = model.blocks[i]

        # attention qkv/proj
        if hasattr(blk.attn, "qkv") and isinstance(blk.attn.qkv, nn.Linear):
            blk.attn.qkv = LoRALinear(blk.attn.qkv, r=r, alpha=alpha, dropout=dropout)
            lora_params += list(blk.attn.qkv.lora_down.parameters()) + list(blk.attn.qkv.lora_up.parameters())

        if hasattr(blk.attn, "proj") and isinstance(blk.attn.proj, nn.Linear):
            blk.attn.proj = LoRALinear(blk.attn.proj, r=r, alpha=alpha, dropout=dropout)
            lora_params += list(blk.attn.proj.lora_down.parameters()) + list(blk.attn.proj.lora_up.parameters())

        # MLP fc1/fc2
        if hasattr(blk.mlp, "fc1") and isinstance(blk.mlp.fc1, nn.Linear):
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, r=r, alpha=alpha, dropout=dropout)
            lora_params += list(blk.mlp.fc1.lora_down.parameters()) + list(blk.mlp.fc1.lora_up.parameters())

        if hasattr(blk.mlp, "fc2") and isinstance(blk.mlp.fc2, nn.Linear):
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, r=r, alpha=alpha, dropout=dropout)
            lora_params += list(blk.mlp.fc2.lora_down.parameters()) + list(blk.mlp.fc2.lora_up.parameters())
            
    
    for p in lora_params:
        p.requres_grad = True
    
    assert len(lora_params) > 0, "No LoRA parameters found. Check the model structure."

    return lora_params
