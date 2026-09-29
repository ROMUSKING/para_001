"""Frozen visual encoders used as model inputs and as the shared evaluation target.

* :class:`ResNet18Embedder` reproduces the pilot's ``embed_images`` exactly (torchvision
  ImageNet ResNet-18, 128 x 128 antialiased resize, ImageNet normalisation, BF16 autocast on
  CUDA, float16 output). Its 512-d global features are the pilot's inputs and targets.
* :class:`DinoV2Embedder` gives DINO-WM-style features: ``dinov2_vits14`` (code and weights
  Apache-2.0, loaded through ``torch.hub``) CLS token plus the ``x_norm_patchtokens`` grid
  average-pooled to ``grid x grid`` tokens.

Both download weights on first use (Colab). Nothing here trains.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _prepare(images_uint8, image_size: int, device) -> torch.Tensor:
    batch = torch.from_numpy(np.stack(images_uint8)).to(device)
    batch = batch.permute(0, 3, 1, 2).contiguous()
    from torchvision.transforms import functional as TVF  # noqa: PLC0415

    batch = TVF.resize(batch, [image_size, image_size], antialias=True)
    batch = batch.float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _autocast(device):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib  # noqa: PLC0415

    return contextlib.nullcontext()


def pool_patch_grid(patch_tokens: torch.Tensor, grid: int) -> torch.Tensor:
    """``[N, P, D]`` square patch grid -> ``[N, grid*grid, D]`` by adaptive average pooling."""
    n, p, d = patch_tokens.shape
    side = int(round(p ** 0.5))
    if side * side != p:
        raise ValueError(f"patch tokens do not form a square grid: {p}")
    x = patch_tokens.transpose(1, 2).reshape(n, d, side, side)
    x = F.adaptive_avg_pool2d(x, grid)
    return x.flatten(2).transpose(1, 2)


class ResNet18Embedder:
    name = "resnet18"

    def __init__(self, device, image_size: int = 128, batch_size: int = 64, weights: str | None = "DEFAULT"):
        from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415

        self.device, self.image_size, self.batch_size = torch.device(device), image_size, batch_size
        resolved = getattr(ResNet18_Weights, weights) if weights else None
        self.weights = str(resolved) if resolved else None
        model = resnet18(weights=resolved)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = 512

    @torch.inference_mode()
    def __call__(self, images_uint8) -> np.ndarray:
        features = []
        for start in range(0, len(images_uint8), self.batch_size):
            batch = _prepare(images_uint8[start : start + self.batch_size], self.image_size, self.device)
            with _autocast(self.device):
                encoded = self.model(batch)
            features.append(encoded.float().cpu())
        return torch.cat(features, dim=0).numpy().astype(np.float16)

    def manifest(self) -> dict:
        return {"name": self.name, "weights": self.weights, "image_size": self.image_size, "dim": self.dim,
                "tokens_per_camera": 1, "licence": "torchvision (BSD-3-Clause); ImageNet weights"}


class DinoV2Embedder:
    name = "dinov2_vits14"

    def __init__(self, device, image_size: int = 224, grid: int = 2, batch_size: int = 64, model_name: str = "dinov2_vits14"):
        if image_size % 14:
            raise ValueError("DINOv2 needs an image size divisible by the 14-pixel patch")
        self.device, self.image_size, self.grid, self.batch_size = torch.device(device), image_size, grid, batch_size
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = int(self.model.embed_dim)

    @torch.inference_mode()
    def __call__(self, images_uint8) -> np.ndarray:
        out = []
        for start in range(0, len(images_uint8), self.batch_size):
            batch = _prepare(images_uint8[start : start + self.batch_size], self.image_size, self.device)
            with _autocast(self.device):
                feats = self.model.forward_features(batch)
            cls = feats["x_norm_clstoken"].float().unsqueeze(1)
            patches = pool_patch_grid(feats["x_norm_patchtokens"].float(), self.grid)
            out.append(torch.cat([cls, patches], dim=1).cpu())
        return torch.cat(out, dim=0).numpy().astype(np.float16)

    def manifest(self) -> dict:
        return {"name": self.name, "hub": f"facebookresearch/dinov2:{self.model_name}", "image_size": self.image_size,
                "dim": self.dim, "tokens_per_camera": 1 + self.grid * self.grid, "grid": self.grid,
                "licence": "Apache-2.0 (code and weights, per the DINOv2 README)"}
