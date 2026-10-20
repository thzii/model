"""Evaluation feature extraction: ℓ2-normalised backbone CLS features (spec §7 step 1)."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from planora_sf.models.backbone import BackboneAdapter


@torch.no_grad()
def extract_features(backbone: BackboneAdapter, batches: Iterable[torch.Tensor]) -> torch.Tensor:
    """Run ``backbone`` in eval mode over image batches and return ℓ2-normalised CLS features.

    The caller supplies test-transform batches (Resize 256 -> CenterCrop 224 ->
    Normalize). A real data loader is a Stage C task.
    """
    was_training = backbone.training
    backbone.eval()
    feats = [F.normalize(backbone(images), dim=-1) for images in batches]
    backbone.train(was_training)
    return torch.cat(feats) if feats else torch.empty(0, backbone.feat_dim)
