import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _effective_number_weights(cls_counts: np.ndarray, beta: float) -> np.ndarray:
    """Class-balanced weights from 'Class-Balanced Loss Based on Effective Number of Samples'."""
    cls_counts = np.asarray(cls_counts, dtype=np.float64).clip(1.0, None)
    effective_num = 1.0 - np.power(beta, cls_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    # Normalize to mean 1.0 (helps LR stability)
    weights = weights / (weights.mean() + 1e-12)
    return weights

class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.
    - Cui et al., 2019 (CVPR)
    """
    def __init__(self, cls_counts: np.ndarray, beta: float = 0.9999, gamma: float = 1.5):
        super().__init__()
        w = _effective_number_weights(cls_counts, beta)
        self.register_buffer("class_weights", torch.tensor(w, dtype=torch.float))
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)                               # [B, C]
        probs = log_probs.exp()
        pt = probs[torch.arange(logits.size(0), device=logits.device), targets]# [B]
        logpt = log_probs[torch.arange(logits.size(0), device=logits.device), targets]
        w = self.class_weights[targets]                                        # [B]
        loss = - w * ((1.0 - pt).clamp_(min=1e-6) ** self.gamma) * logpt
        return loss.mean()

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (LDAM) with optional DRW (Deferred Re-Weighting).
    - Cao et al., 2019 (NeurIPS)
    """
    def __init__(self,
                 cls_counts: np.ndarray,
                 max_m: float = 0.5,
                 s: float = 30.0,
                 drw: bool = True,
                 beta: float = 0.9999,
                 drw_start: int = 15):
        super().__init__()
        cls_counts = np.asarray(cls_counts, dtype=np.float64).clip(1.0, None)
        # margins m_c = m_max / n_c^{1/4}
        m_list = max_m / np.power(cls_counts, 0.25)
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float))
        self.s = float(s)
        self.drw = bool(drw)
        self.drw_start = int(drw_start)
        # DRW weights (effective-number)
        w = _effective_number_weights(cls_counts, beta)
        self.register_buffer("class_weights", torch.tensor(w, dtype=torch.float))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        # Subtract margin on the true class logits
        margins = self.m_list[targets]  # [B]
        margins = margins.to(logits.dtype) 
        x = logits.clone()
        idx = torch.arange(x.size(0), device=x.device)
        x[idx, targets] = x[idx, targets] - margins
        # Scale then CE with optional DRW weights
        weight = None
        if self.drw and (epoch is not None) and (epoch >= self.drw_start):
            weight = self.class_weights
        return F.cross_entropy(self.s * x, targets, weight=weight)
