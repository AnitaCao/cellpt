# utils/targeted_aug.py
from __future__ import annotations
import os, random, math
from dataclasses import dataclass
from typing import Dict, List, Iterable
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T

@dataclass
class PolicyCfg:
    # Probabilities to apply each transform (per sample)
    p_geom: float = 0.85          # flip/affine
    p_color: float = 0.65         # brightness/contrast/saturation/hue
    p_blur: float  = 0.30
    p_erase: float = 0.30
    # Geometric magnitudes
    max_rotate: float = 12.0      # degrees
    max_shear: float  = 10.0
    max_translate: float = 0.04   # fraction of H/W
    min_scale: float = 0.90
    max_scale: float = 1.10
    # Photometric magnitudes
    b_delta: float = 0.25
    c_delta: float = 0.25
    s_delta: float = 0.15
    h_delta: float = 0.05
    # Blur/erasing
    blur_sigma: tuple[float, float] = (0.1, 1.0)
    erase_scale: tuple[float, float] = (0.02, 0.12)
    erase_ratio: tuple[float, float] = (0.3, 3.3)

def _to_three(x):
    # x: tensor [..., C, H, W]; return (C,H,W)
    return x

class ClassAwareAugmenter:
    """
    Class-aware, view-synchronized augmenter.
    - x: [B,C,H,W] or [B,S,C,H,W] (float tensor, already normalized)
    - y: [B] integer labels (hard)
    """
    def __init__(
        self,
        class_to_idx: Dict[str, int],
        mean: Iterable[float],
        std: Iterable[float],
        weak_class_names: List[str],
        # policies
        weak_morph: PolicyCfg | None = None,
        weak_color: PolicyCfg | None = None,
        default_policy: PolicyCfg | None = None,
        # knobs
        same_color_across_views: bool = True,
        same_geom_across_views: bool = True,
    ):
        self.class_to_idx = class_to_idx
        self.mean = torch.tensor(list(mean)).view(1, -1, 1, 1)
        self.std  = torch.tensor(list(std)).view(1, -1, 1, 1)
        weak_idxs = [class_to_idx[n] for n in weak_class_names if n in class_to_idx]
        self.weak_idx_set = set(weak_idxs)

        # policies
        self.policy_default = default_policy or PolicyCfg(
            p_geom=0.25, p_color=0.10, p_blur=0.05, p_erase=0.10,
            max_rotate=8.0, max_shear=6.0, max_translate=0.02,
            min_scale=0.95, max_scale=1.05,
            b_delta=0.10, c_delta=0.10, s_delta=0.05, h_delta=0.02,
            blur_sigma=(0.1,0.6), erase_scale=(0.02,0.08),
        )
        # heavier for morphology-centric errors (Endothelial, Myeloid)
        self.policy_weak_morph = weak_morph or PolicyCfg(
            p_geom=0.90, p_color=0.35, p_blur=0.35, p_erase=0.30,
            max_rotate=12.0, max_shear=10.0, max_translate=0.04,
            min_scale=0.90, max_scale=1.10,
            b_delta=0.10, c_delta=0.15, s_delta=0.10, h_delta=0.02,
            blur_sigma=(0.1,1.0), erase_scale=(0.02,0.12),
        )
        # heavier for color/stain sensitivity (T cells)
        self.policy_weak_color = weak_color or PolicyCfg(
            p_geom=0.65, p_color=0.85, p_blur=0.30, p_erase=0.25,
            max_rotate=10.0, max_shear=8.0, max_translate=0.03,
            min_scale=0.92, max_scale=1.08,
            b_delta=0.25, c_delta=0.25, s_delta=0.15, h_delta=0.05,
            blur_sigma=(0.1,1.0), erase_scale=(0.02,0.10),
        )

        self.same_color_across_views = same_color_across_views
        self.same_geom_across_views = same_geom_across_views

        # modules that are happy on GPU tensors
        self.eraser = T.RandomErasing(p=1.0, scale=self.policy_default.erase_scale,
                                      ratio=self.policy_default.erase_ratio, value=0)

    def _unnorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def _renorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _sample_geom(self, pol: PolicyCfg, H: int, W: int):
        angle = random.uniform(-pol.max_rotate, pol.max_rotate)
        shear = random.uniform(-pol.max_shear, pol.max_shear)
        tx = int(round(random.uniform(-pol.max_translate, pol.max_translate) * W))
        ty = int(round(random.uniform(-pol.max_translate, pol.max_translate) * H))
        scale = random.uniform(pol.min_scale, pol.max_scale)
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.1
        return angle, shear, (tx, ty), scale, flip_h, flip_v

    def _apply_geom(self, img: torch.Tensor, angle, shear, translate, scale, flip_h, flip_v):
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=[shear, 0.0],
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        if flip_h: img = TF.hflip(img)
        if flip_v: img = TF.vflip(img)
        return img

    def _apply_color(self, img: torch.Tensor, pol: PolicyCfg):
        # unnorm -> jitter -> clamp -> renorm
        img = self._unnorm(img)
        # brightness, contrast, saturation, hue
        b = 1.0 + random.uniform(-pol.b_delta, pol.b_delta)
        c = 1.0 + random.uniform(-pol.c_delta, pol.c_delta)
        s = 1.0 + random.uniform(-pol.s_delta, pol.s_delta)
        h = random.uniform(-pol.h_delta, pol.h_delta)
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        img = TF.adjust_saturation(img, s)
        img = TF.adjust_hue(img, h)
        img = img.clamp(0.0, 1.0)
        img = self._renorm(img)
        return img

    def _maybe_blur(self, img: torch.Tensor, pol: PolicyCfg):
        sigma = random.uniform(*pol.blur_sigma)
        # kernel size 3 or 5 based on sigma
        k = 3 if sigma < 0.8 else 5
        return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)

    def _maybe_erase(self, img: torch.Tensor, pol: PolicyCfg):
        # reuse the same module; it samples its own box
        return self.eraser(img)

    def _policy_for(self, class_idx: int) -> PolicyCfg:
        # heuristic: if the class name contains "T cell", prefer color policy
        # otherwise default to morph policy for weak classes
        # (feel free to swap mapping in your config)
        if class_idx in self.weak_idx_set:
            # lightweight heuristic hook (string check)
            return self.policy_weak_color if "t cell" in self._name_for(class_idx).lower() else self.policy_weak_morph
        return self.policy_default

    def _name_for(self, idx: int) -> str:
        for k, v in self.class_to_idx.items():
            if v == idx: return k
        return str(idx)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply in-place‑safe transforms and return augmented tensor.
        """
        orig_dtype = x.dtype
        if x.ndim == 4:
            B, C, H, W = x.shape
            S = 1
        elif x.ndim == 5:
            B, S, C, H, W = x.shape
        else:
            return x

        # Work on the same device as x
        out = x

        for i in range(B):
            cls = int(y[i].item())
            pol = self._policy_for(cls)
            do_geom  = (random.random() < pol.p_geom)
            do_color = (random.random() < pol.p_color)
            do_blur  = (random.random() < pol.p_blur)
            do_erase = (random.random() < pol.p_erase)

            # Sample once per sample (and share across views if requested)
            if do_geom:
                angle, shear, trans, scale, fh, fv = self._sample_geom(pol, H, W)

            for s in range(S):
                img = out[i, s] if S > 1 else out[i]

                if do_geom:
                    if not self.same_geom_across_views and S > 1 and s > 0:
                        angle, shear, trans, scale, fh, fv = self._sample_geom(pol, H, W)
                    img = self._apply_geom(img, angle, shear, trans, scale, fh, fv)

                if do_color:
                    img = self._apply_color(img, pol)  # handles (un)normalize

                if do_blur:
                    img = self._maybe_blur(img, pol)

                if do_erase:
                    img = self._maybe_erase(img, pol)

                if S > 1:
                    out[i, s] = img
                else:
                    out[i] = img

        return out.to(dtype=orig_dtype)
