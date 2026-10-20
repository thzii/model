"""PLANORA-SF network: backbone -> (DINOHead -> ℓ2 = z, linear L_cls head)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from planora_sf.config import PlanoraSFConfig
from planora_sf.models.backbone import BackboneAdapter, build_backbone
from planora_sf.models.cls_head import LinearClsHead
from planora_sf.models.projection_head import DINOHead


@dataclass
class ModelOutput:
    cls: torch.Tensor  # [B, F] backbone CLS features, un-normalised (used for clustering after ℓ2)
    z: torch.Tensor  # [B, D] ℓ2-normalised contrastive embedding
    cls_logits: Optional[torch.Tensor]  # [B, C] logits of the L_cls head, if enabled


class PlanoraSFModel(nn.Module):
    def __init__(self, backbone: BackboneAdapter, head: DINOHead, cls_head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.cls_head = cls_head

    @property
    def latent_dim(self) -> int:
        return self.head.last_layer.out_features

    def forward(self, x: torch.Tensor) -> ModelOutput:
        cls = self.backbone(x)
        z = F.normalize(self.head(cls), dim=-1)
        logits = self.cls_head(cls) if self.cls_head is not None else None
        return ModelOutput(cls=cls, z=z, cls_logits=logits)


def build_model(
    cfg: PlanoraSFConfig,
    num_known_classes: int,
    *,
    backbone: Optional[BackboneAdapter] = None,
    pretrained: bool = True,
) -> PlanoraSFModel:
    backbone = backbone if backbone is not None else build_backbone(cfg.backbone, pretrained=pretrained)
    head = DINOHead(
        backbone.feat_dim,
        cfg.latent.dim,
        hidden_dim=cfg.head.hidden_dim,
        bottleneck_dim=cfg.head.bottleneck_dim,
        num_layers=cfg.head.num_layers,
    )
    cls_head = None
    if cfg.cls_head.enabled:
        if cfg.cls_head.input != "cls":
            raise NotImplementedError("U-17: only the CLS-input classification head is scaffolded")
        cls_head = LinearClsHead(backbone.feat_dim, cfg.cls_head.num_classes or num_known_classes)
    return PlanoraSFModel(backbone, head, cls_head)
