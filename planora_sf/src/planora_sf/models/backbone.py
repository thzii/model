"""Backbone adapter: wraps a ViT that returns its CLS token and controls which blocks train.

Feature dimension is read from the model (``embed_dim`` / ``num_features``) or
probed with a dummy forward pass; it is never hard-coded (spec §3.2 C3-C4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch import nn

from planora_sf.config import BackboneConfig
from planora_sf.models.tiny_vit import TinyViT

# torch.hub entry points for the backbones named in FX-P §5.1. Loading them needs
# network access (or a populated hub cache) and is not exercised by the unit tests.
HUB_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "dinov1_vitb16": ("facebookresearch/dino:main", "dino_vitb16"),
    "dinov2_vitb14_reg": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
}


def infer_feature_dim(vit: nn.Module, probe_image_size: Optional[int] = None) -> int:
    for attr in ("embed_dim", "num_features"):
        value = getattr(vit, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    if probe_image_size is None:
        raise ValueError("cannot infer the feature dimension: the model has no embed_dim/num_features "
                         "attribute and no probe_image_size was given")
    param = next(vit.parameters())
    was_training = vit.training
    vit.eval()
    with torch.no_grad():
        out = vit(torch.zeros(1, 3, probe_image_size, probe_image_size, device=param.device, dtype=param.dtype))
    vit.train(was_training)
    if out.ndim != 2:
        raise ValueError(f"backbone must return [B, F] CLS features, got shape {tuple(out.shape)}")
    return int(out.shape[1])


class BackboneAdapter(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        *,
        blocks_attr: str = "blocks",
        feat_dim: Optional[int] = None,
        probe_image_size: Optional[int] = None,
    ):
        super().__init__()
        if not isinstance(getattr(vit, blocks_attr, None), (nn.ModuleList, nn.Sequential)):
            raise ValueError(f"backbone has no '{blocks_attr}' ModuleList")
        self.vit = vit
        self.blocks_attr = blocks_attr
        self.feat_dim = feat_dim if feat_dim is not None else infer_feature_dim(vit, probe_image_size)

    @property
    def blocks(self) -> nn.ModuleList:
        return getattr(self.vit, self.blocks_attr)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vit(x)
        if out.ndim != 2 or out.shape[1] != self.feat_dim:
            raise ValueError(f"expected CLS features [B, {self.feat_dim}], got {tuple(out.shape)}")
        return out

    def freeze(self, grad_from_block: int, *, train_final_norm: bool = False) -> None:
        """Freeze everything, then unfreeze blocks ``grad_from_block ..`` (SelEx ct.py:746-758).

        The SelEx code only unfreezes parameters whose name contains "block", so the
        final norm stays frozen unless ``train_final_norm`` is set.
        """
        if not 0 <= grad_from_block <= self.num_blocks:
            raise ValueError(f"grad_from_block must be in [0, {self.num_blocks}], got {grad_from_block}")
        for param in self.vit.parameters():
            param.requires_grad_(False)
        for index, block in enumerate(self.blocks):
            if index >= grad_from_block:
                for param in block.parameters():
                    param.requires_grad_(True)
        norm = getattr(self.vit, "norm", None)
        if train_final_norm and isinstance(norm, nn.Module):
            for param in norm.parameters():
                param.requires_grad_(True)

    def trainable_block_indices(self) -> list[int]:
        return [i for i, block in enumerate(self.blocks) if any(p.requires_grad for p in block.parameters())]


def strip_dino_checkpoint(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Turn a DINO training checkpoint into backbone weights, as SelEx does (ct.py:718-724):
    take the ``teacher`` entry when present and drop the ``backbone.`` prefix and ``head.`` keys."""
    if "teacher" in state:
        state = state["teacher"]
    out = {}
    for key, value in state.items():
        key = key[len("backbone."):] if key.startswith("backbone.") else key
        if key.startswith("head."):
            continue
        out[key] = value
    return out


def build_backbone(
    cfg: BackboneConfig,
    *,
    pretrained: bool = True,
    checkpoint_path: Optional[Union[str, Path]] = None,
    tiny_kwargs: Optional[dict[str, Any]] = None,
) -> BackboneAdapter:
    """Build and freeze the configured backbone.

    ``"tiny"`` builds the synthetic test stand-in. The DINO backbones are fetched
    through torch.hub (network access needed), optionally followed by loading a
    local checkpoint.
    """
    if cfg.name == "tiny":
        vit = TinyViT(img_size=cfg.image_size, **(tiny_kwargs or {}))
    else:
        repo, entry = HUB_ENTRYPOINTS[cfg.name]
        vit = torch.hub.load(repo, entry, pretrained=pretrained and checkpoint_path is None)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            vit.load_state_dict(strip_dino_checkpoint(state))
    adapter = BackboneAdapter(vit, probe_image_size=cfg.image_size)
    adapter.freeze(cfg.grad_from_block, train_final_norm=cfg.train_final_norm)
    return adapter
