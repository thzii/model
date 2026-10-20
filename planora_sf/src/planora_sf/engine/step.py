"""One loss evaluation on one mini-batch (spec §6, part B).

This computes losses for a given batch so that gradient flow can be checked.
It is not a training loop: no data loading, no epochs, no checkpointing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from planora_sf.config import PlanoraSFConfig
from planora_sf.fourier.views import FourierViewGenerator, FourierViews
from planora_sf.latent.partition import high_width, low_width
from planora_sf.losses.contrastive import ContrastiveSettings
from planora_sf.losses.fourex import cls_loss, high_loss, low_loss
from planora_sf.losses.gcd import gcd_loss
from planora_sf.losses.selex import selex_loss
from planora_sf.losses.total import LossBreakdown, combine_losses
from planora_sf.models.model import PlanoraSFModel
from planora_sf.pseudo_labels.hssk import PseudoLabelBatch, PseudoLabelState, build_hierarchy
from planora_sf.pseudo_labels.target_matrix import build_use_targets


@dataclass
class TrainBatch:
    view1: torch.Tensor  # [B, C, H, W] first augmented view
    view2: torch.Tensor  # [B, C, H, W] second augmented view
    targets: torch.Tensor  # [B] known-class index for labelled samples, -1 otherwise
    labelled: torch.Tensor  # [B] bool

    def __post_init__(self) -> None:
        if self.view1.shape != self.view2.shape:
            raise ValueError("view1 and view2 must have the same shape")
        batch = self.view1.shape[0]
        if self.targets.shape != (batch,) or self.labelled.shape != (batch,):
            raise ValueError("targets and labelled must have shape [B]")
        if (self.targets[self.labelled.bool()] < 0).any():
            raise ValueError("labelled samples need a non-negative target")


def refresh_pseudo_labels(
    features: torch.Tensor,
    labelled_mask: torch.Tensor,
    targets: torch.Tensor,
    num_known: int,
    num_novel: int,
    cfg: PlanoraSFConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> PseudoLabelState:
    """Spec §6, part A, given precomputed ℓ2-normalised CLS features of view 0 of every training image.

    Extracting those features from a dataset is a Stage C task.
    """
    levels = build_hierarchy(
        features, labelled_mask, targets, num_known, num_novel,
        balanced=cfg.selex.balanced, max_iter=cfg.selex.kmeans_max_iter, n_init=cfg.selex.kmeans_n_init,
        reproduce_label_order_bug=cfg.selex.reproduce_label_order_bug, generator=generator,
    )
    return PseudoLabelState(levels)


def compute_losses(
    model: PlanoraSFModel,
    fourier: FourierViewGenerator,
    batch: TrainBatch,
    pseudo: PseudoLabelBatch,
    cfg: PlanoraSFConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[LossBreakdown, FourierViews]:
    """Forward the 4-view batch [x(1); x(2); l; h] and evaluate L_total (FX-P Eq. 9)."""
    if not cfg.fourier.extra_views:
        raise NotImplementedError("U-01: the 'substitute x(2)' variant is not scaffolded (Stage C)")
    if cfg.fourier.filtered_view != "x1":
        raise NotImplementedError("U-01: only filtering x(1) is scaffolded")
    if cfg.fourier.blur_anchor:
        raise NotImplementedError("U-03: a blurred anchor needs a fifth forward pass; not scaffolded")

    settings = ContrastiveSettings.from_config(cfg)
    lam, alpha, loss_type = cfg.selex.lam, cfg.selex.smoothing_alpha, cfg.selex.use_loss
    targets, labelled = batch.targets, batch.labelled.bool()

    views = fourier(batch.view1, generator=generator)
    out = model(torch.cat([batch.view1, batch.view2, views.low, views.high]))
    z1, z2, z_low, z_high = out.z.chunk(4)
    c1, c2, c_low, _ = out.cls.chunk(4)
    pseudo2 = pseudo.repeat_views(2)
    mode = cfg.resolved_target_mode

    if cfg.base == "selex":
        base_targets = build_use_targets(mode, pseudo2, backbone_feats=torch.cat([c1, c2]))
        base = selex_loss(
            z1, z2, targets, labelled, pseudo.level_labels, base_targets,
            lam=lam, alpha=alpha, settings=settings, loss_type=loss_type,
        ).total
    else:
        base = gcd_loss(z1, z2, targets, labelled, lam=lam, settings=settings)

    dim = out.z.shape[1]
    half = cfg.latent.fraction_of_half
    d_low = low_width(views.f_low.cpu(), cfg.fourier.t_low(), dim, fraction_of_half=half)
    d_high = high_width(
        views.f_high.cpu(), cfg.fourier.t_high(), dim, mode=cfg.latent.high_fraction_mode, fraction_of_half=half
    )

    low_targets = build_use_targets(mode, pseudo2, backbone_feats=torch.cat([c1, c_low]))
    low = low_loss(
        z1, z_low, d_low, targets, labelled, pseudo.level_labels, low_targets,
        lam=lam, alpha=alpha, settings=settings, loss_type=loss_type,
    )
    high = high_loss(z1, z_high, d_high, targets, labelled, lam=lam, settings=settings)

    if out.cls_logits is not None:
        logits_x, _, logits_low, logits_high = out.cls_logits.chunk(4)
        cls = cls_loss([logits_high, logits_low, logits_x], targets, labelled, kind=cfg.cls_head.loss)
    else:
        cls = out.z.sum() * 0.0

    return combine_losses(base, low, high, cls, cfg.weights), views
