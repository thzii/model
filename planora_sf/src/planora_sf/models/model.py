"""PLANORA-SF network: backbone -> (DINOHead -> ℓ2 or Poincaré ball = z, linear L_cls head, optional DebGCD heads)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from planora_sf.config import PlanoraSFConfig
from planora_sf.debgcd.heads import DebGCDHeads
from planora_sf.hyperbolic.layers import ToPoincare
from planora_sf.models.backbone import BackboneAdapter, build_backbone
from planora_sf.models.cls_head import LinearClsHead
from planora_sf.models.projection_head import DINOHead


@dataclass
class ModelOutput:
    cls: torch.Tensor  # [B, F] backbone CLS features, un-normalised (used for clustering after ℓ2)
    z: torch.Tensor  # [B, D] contrastive embedding: ℓ2-normalised, or a Poincaré-ball point under HypCD
    cls_logits: Optional[torch.Tensor]  # [B, C] logits of the L_cls head, if enabled
    z_tangent: Optional[torch.Tensor] = None  # [B, D] head output before ℓ2 / clipping + expmap0


class PlanoraSFModel(nn.Module):
    def __init__(
        self,
        backbone: BackboneAdapter,
        head: DINOHead,
        cls_head: Optional[nn.Module] = None,
        debgcd: Optional[DebGCDHeads] = None,
        to_ball: Optional[ToPoincare] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.cls_head = cls_head
        self.debgcd = debgcd  # applied by engine.step to the two standard views only (plan C4)
        self.to_ball = to_ball  # HypCD: replaces the ℓ2 normalisation of z (upstream train_HypSelEx.py:275)

    @property
    def latent_dim(self) -> int:
        return self.head.last_layer.out_features

    def forward(self, x: torch.Tensor) -> ModelOutput:
        cls = self.backbone(x)
        v = self.head(cls)
        z = self.to_ball(v) if self.to_ball is not None else F.normalize(v, dim=-1)
        logits = self.cls_head(cls) if self.cls_head is not None else None
        return ModelOutput(cls=cls, z=z, cls_logits=logits, z_tangent=v)


def build_model(
    cfg: PlanoraSFConfig,
    num_known_classes: int,
    *,
    backbone: Optional[BackboneAdapter] = None,
    pretrained: bool = True,
    num_classes: Optional[int] = None,
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
    debgcd = None
    if cfg.debgcd.enabled:  # built last, so the existing heads' initialisation draws are unchanged
        if num_classes is None:
            raise ValueError("DebGCD needs num_classes (known + novel) for the debiased classifier")
        debgcd = DebGCDHeads(backbone.feat_dim, num_known_classes, num_classes, cfg.debgcd)
    to_ball = None
    if cfg.hypcd.enabled:  # parameter-free, so no initialisation draws or state-dict keys change
        if cfg.base != "selex":
            raise NotImplementedError("HypCD is ported for the SelEx base only (Hyp-SelEx); Hyp-GCD is out of scope")
        if cfg.latent.renormalize_slices:
            raise ValueError("HypCD needs latent.renormalize_slices=False: slices of ball points are not re-normalised")
        to_ball = ToPoincare(cfg.hypcd.curvature, cfg.hypcd.clip_r, cfg.hypcd.project_eps)
    return PlanoraSFModel(backbone, head, cls_head, debgcd, to_ball)
