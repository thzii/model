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


def hub_source(name: str) -> Optional[str]:
    """``"repo:ref/entry"`` of a hub backbone, for run provenance; ``None`` for the tiny test ViT."""
    if name not in HUB_ENTRYPOINTS:
        return None
    repo, entry = HUB_ENTRYPOINTS[name]
    return f"{repo}/{entry}"


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


# Foreign checkpoints (official SelEx/SimGCD/DebGCD/HypCD code) wrap or prefix the ViT weights:
# nn.Sequential(backbone, head) saves "0.blocks...", DDP adds "module.", DINO adds "backbone.".
_FOREIGN_WRAPPERS = ("state_dict", "model", "teacher", "student", "backbone")
_FOREIGN_PREFIXES = ("module.", "backbone.", "vit.", "0.")


def _is_state_dict(obj: Any) -> bool:
    return isinstance(obj, Mapping) and bool(obj) and all(isinstance(v, torch.Tensor) for v in obj.values())


def foreign_backbone_state(obj: Any, target_keys: set[str]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Map a foreign checkpoint onto the ViT's own key names (master plan Task #2a re-scoring).

    Unwraps ``state_dict``/``model``/``teacher``/``student``/``backbone`` containers (or a pickled
    module) and strips the prefixes above until a key matches ``target_keys``. Keys that never
    match (heads, projectors) are ignored and counted. Returns ``(state, report)``.
    """
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    while isinstance(obj, Mapping) and not _is_state_dict(obj):
        inner = next((obj[k] for k in _FOREIGN_WRAPPERS if k in obj), None)
        if inner is None:
            raise ValueError(f"no state dict found in the checkpoint (top-level keys: {sorted(obj)[:10]})")
        obj = inner.state_dict() if isinstance(inner, nn.Module) else inner
    if not _is_state_dict(obj):
        raise ValueError(f"expected a mapping of tensors, got {type(obj).__name__}")
    state, ignored = {}, []
    for key, value in obj.items():
        name = key
        while name not in target_keys:
            prefix = next((p for p in _FOREIGN_PREFIXES if name.startswith(p)), None)
            if prefix is None:
                break
            name = name[len(prefix):]
        if name in target_keys:
            if name in state:
                raise ValueError(f"two checkpoint keys map to backbone key {name!r}")
            state[name] = value
        else:
            ignored.append(key)
    return state, {
        "loaded": len(state), "num_ignored": len(ignored), "ignored_examples": sorted(ignored)[:10],
        "missing": sorted(target_keys - set(state)),
    }


def load_foreign_backbone(vit: nn.Module, obj: Any) -> dict[str, Any]:
    """Load foreign ViT weights into ``vit`` strictly: every backbone tensor must be present."""
    state, report = foreign_backbone_state(obj, set(vit.state_dict()))
    if report["missing"]:
        raise ValueError(f"{len(report['missing'])} backbone tensors missing from the checkpoint, "
                         f"e.g. {report['missing'][:5]}; ignored keys look like {report['ignored_examples'][:5]}")
    vit.load_state_dict(state, strict=True)
    return report


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
