"""One loss evaluation on one mini-batch (spec §6, part B).

This computes losses for a given batch so that gradient flow can be checked.
It is not a training loop: no data loading, no epochs, no checkpointing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from planora_sf.config import PlanoraSFConfig
from planora_sf.debgcd.losses import debgcd_terms
from planora_sf.debgcd.surrogate import centroid_posterior
from planora_sf.fourier.views import FourierViewGenerator, FourierViews
from planora_sf.latent.partition import high_width, low_width
from planora_sf.losses.contrastive import ContrastiveSettings
from planora_sf.losses.fourex import cls_loss, high_loss, low_loss
from planora_sf.losses.gcd import gcd_loss
from planora_sf.losses.selex import selex_loss
from planora_sf.losses.total import LossBreakdown, add_debgcd, combine_losses
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
    """Forward the 4-view batch [x(1); x(2); l; h] and evaluate L_total (FX-P Eq. 9).

    With ``cfg.debgcd.enabled`` this adds λ_sdl L_sdl + λ_adl L_adl (DebGCD Eq. 14, plan §2.2),
    computed on the CLS of x(1) and x(2) only: never on the Fourier views (plan C4).
    """
    if not cfg.fourier.extra_views:
        raise NotImplementedError("U-01: the 'substitute x(2)' variant is not scaffolded (Stage C)")
    if cfg.fourier.filtered_view != "x1":
        raise NotImplementedError("U-01: only filtering x(1) is scaffolded")
    if cfg.fourier.blur_anchor:
        raise NotImplementedError("U-03: a blurred anchor needs a fifth forward pass; not scaffolded")
    if cfg.hypcd.enabled:
        raise NotImplementedError("HypCD losses are plan Phase 4; z is on the Poincaré ball and needs them")

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

    losses = combine_losses(base, low, high, cls, cfg.weights)
    if cfg.debgcd.enabled:
        losses = _add_debgcd(model, torch.cat([c1, c2]), batch, pseudo, cfg, losses)
    return losses, views


def _add_debgcd(
    model: PlanoraSFModel, cls: torch.Tensor, batch: TrainBatch, pseudo: PseudoLabelBatch, cfg: PlanoraSFConfig,
    losses: LossBreakdown,
) -> LossBreakdown:
    """DebGCD terms on ``cls [2B, F]``; p_i comes from the level-0 BSSK centroids (plan §2.1, C1)."""
    if model.debgcd is None:
        raise ValueError("debgcd.enabled but the model has no DebGCD heads: build it with build_model(cfg, ...)")
    if pseudo.centers is None or pseudo.num_known_clusters is None:
        raise ValueError("DebGCD needs the level-0 centres in the pseudo-label batch")
    dg = cfg.debgcd
    ova_logits, deb_logits = model.debgcd(cls)
    posterior = centroid_posterior(cls.detach(), pseudo.centers, dg.tau_p)
    terms = debgcd_terms(
        ova_logits, deb_logits, posterior, batch.targets.repeat(2), batch.labelled.bool().repeat(2),
        pseudo.num_known_clusters, threshold=dg.threshold, consistency_filter=dg.consistency_filter,
    )
    return add_debgcd(losses, terms.sdl, terms.adl, dg.lam_sdl, dg.lam_adl, {"adl_utilisation": terms.utilisation})
