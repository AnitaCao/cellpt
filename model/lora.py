# model/lora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=4, alpha=16):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Original frozen linear layer
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA trainable adapters
        self.lora_down = nn.Linear(self.in_features, r, bias=False)
        self.lora_up = nn.Linear(r, self.out_features, bias=False)

        # Init
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.scaling * self.lora_up(self.lora_down(x))
