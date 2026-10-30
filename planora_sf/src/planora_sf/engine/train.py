"""Training loop (spec §6), single device.

Each epoch alternates two phases:
  A. no grad: extract ℓ2-normalised CLS features of every training image, then
     run BSSK/HSSK to refresh the pseudo-labels and cluster radii;
  B. contrastive mini-batch steps with the GCD sampler: the 4-view forward pass
     [x(1); x(2); l; h], L_total, backward, then SGD.
The cosine LR schedule steps once per epoch (SelEx ct.py:211-215, 415).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from torch.utils.data import DataLoader

from planora_sf.config import PlanoraSFConfig
from planora_sf.data.factory import GCDDatasets
from planora_sf.data.gcd_dataset import GCDTrainDataset
from planora_sf.data.sampler import build_gcd_sampler
from planora_sf.debgcd.surrogate import align_novel_clusters, permute_level
from planora_sf.engine.diagnostics import pseudo_label_quality
from planora_sf.engine.optim import build_optimizer, build_scheduler
from planora_sf.engine.step import TrainBatch, compute_losses, refresh_pseudo_labels
from planora_sf.eval.metrics import AccuracyReport
from planora_sf.eval.protocol import evaluate_transductive, extract_dataset_features
from planora_sf.fourier.views import FourierViewGenerator
from planora_sf.models.model import PlanoraSFModel
from planora_sf.pseudo_labels.hssk import PseudoLabelState

log = logging.getLogger(__name__)


@dataclass
class EpochStats:
    epoch: int
    steps: int
    losses: dict[str, float]  # mean over steps
    lr: float  # learning rate used during this epoch
    num_levels: int  # pseudo-label hierarchy depth
    seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)  # loop.diagnostics / gradient monitor


@dataclass
class FitResult:
    history: list[EpochStats] = field(default_factory=list)
    evaluations: list[tuple[int, dict[str, AccuracyReport]]] = field(default_factory=list)
    state: Optional[PseudoLabelState] = None


def refresh_from_dataset(
    model: PlanoraSFModel,
    dataset: GCDTrainDataset,
    cfg: PlanoraSFConfig,
    *,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 128,
    generator: Optional[torch.Generator] = None,
) -> PseudoLabelState:
    """Phase A: features of every training image (index order) -> BSSK/HSSK pseudo-labels."""
    feats = extract_dataset_features(
        model.backbone, dataset, batch_size=batch_size, num_workers=cfg.loop.num_workers, device=device
    )
    return refresh_pseudo_labels(
        feats, dataset.is_labelled, dataset.train_targets, dataset.split.num_known, dataset.split.num_novel,
        cfg, generator=generator,
    )


def align_to_previous(state: PseudoLabelState, previous: Optional[PseudoLabelState], num_known: int) -> PseudoLabelState:
    """DebGCD only: keep level-0 novel cluster ids stable across epochs (plan C2).

    Checks that every known class anchors its own level-0 cluster, so that cluster j is
    known class j for the debiased classifier (DG-08).
    """
    base = state.levels[0]
    if base.num_known_clusters != num_known:
        raise ValueError(
            f"DG-08: {base.num_known_clusters} of {num_known} known classes have labelled samples; "
            "level-0 cluster ids would not match known-class targets"
        )
    if previous is None:
        return state
    perm = align_novel_clusters(previous.levels[0].centers, base.centers, num_known)
    return PseudoLabelState([permute_level(base, perm), *state.levels[1:]])


def train_one_epoch(
    model: PlanoraSFModel,
    fourier: FourierViewGenerator,
    loader: DataLoader,
    state: PseudoLabelState,
    optimizer: torch.optim.Optimizer,
    cfg: PlanoraSFConfig,
    *,
    device: Union[str, torch.device] = "cpu",
    max_steps: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    epoch: Optional[int] = None,
) -> tuple[int, dict[str, float]]:
    """Phase B over one pass of ``loader``. Returns ``(steps, mean losses and monitors)``.

    HypCD needs the 0-based ``epoch`` for α_d (the schedule spans ``cfg.optim.epochs``).

    Training targets come from ``dataset.train_targets``, so unlabelled
    ground truth (which the dataset also yields for evaluation) is never used.
    """
    dataset: GCDTrainDataset = loader.dataset
    visible_targets = dataset.train_targets
    hyp_alpha = None
    if cfg.hypcd.enabled:
        if epoch is None:
            raise ValueError("HypCD needs the epoch index for its α_d schedule")
        hyp_alpha = cfg.hypcd.alpha(epoch, cfg.optim.epochs)
    model.train()
    totals: dict[str, float] = defaultdict(float)
    steps = 0
    for views, _, is_labelled, index in loader:
        if max_steps is not None and steps >= max_steps:
            break
        batch = TrainBatch(
            view1=views[0].to(device),
            view2=views[1].to(device),
            targets=visible_targets[index].to(device),
            labelled=is_labelled.to(device),
        )
        losses, _ = compute_losses(
            model, fourier, batch, state.batch(index).to(device), cfg, generator=generator, hyp_alpha=hyp_alpha
        )
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"non-finite loss at step {steps}: {losses.as_floats()}")
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        optimizer.step()
        for name, value in losses.as_floats().items():
            totals[name] += value
        steps += 1
    return steps, {name: value / steps for name, value in totals.items()} if steps else {}


def save_checkpoint(
    path: Path, model, optimizer, scheduler, epoch: int, cfg: PlanoraSFConfig,
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg.to_dict(),
            "provenance": dict(provenance or {}),
        },
        path,
    )


def fit(
    model: PlanoraSFModel,
    datasets: GCDDatasets,
    cfg: PlanoraSFConfig,
    *,
    device: Union[str, torch.device] = "cpu",
    epochs: Optional[int] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    generator: Optional[torch.Generator] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> FitResult:
    """Run ``epochs`` (default ``cfg.optim.epochs``) of alternating optimisation.

    The LR schedule always spans ``cfg.optim.epochs``, so a shorter dry run ends
    mid-schedule. ``last.pt`` is overwritten every epoch; ``loop.keep_checkpoint_every``
    also keeps ``epoch_XXX.pt`` copies for offline learning curves. Selecting the best
    checkpoint on test labels (D-12) is deliberately not done (U-28). ``provenance``
    (seed, commit, hashes; see ``engine.provenance``) is stored in every checkpoint.
    """
    device = torch.device(device)
    model.to(device)
    num_epochs = cfg.optim.epochs if epochs is None else epochs
    train = datasets.train
    if len(train) < cfg.optim.batch_size:
        raise ValueError(f"training set ({len(train)}) is smaller than one batch ({cfg.optim.batch_size})")

    fourier = FourierViewGenerator(cfg.fourier)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim)
    sampler = build_gcd_sampler(train.num_labelled, train.num_unlabelled, generator=generator)
    loader = DataLoader(
        train, batch_size=cfg.optim.batch_size, sampler=sampler, drop_last=True, num_workers=cfg.loop.num_workers
    )
    prototype_dataset = train if cfg.loop.prototype_features == "train_view0" else datasets.train_eval

    result = FitResult()
    previous: Optional[PseudoLabelState] = None
    for epoch in range(num_epochs):
        start = time.perf_counter()
        state = refresh_from_dataset(  # phase A
            model, prototype_dataset, cfg, device=device, batch_size=cfg.optim.batch_size, generator=generator
        )
        if cfg.debgcd.enabled:
            state = align_to_previous(state, previous if cfg.debgcd.align_clusters else None, train.split.num_known)
            previous = state
        diagnostics: dict[str, Any] = {}
        if cfg.loop.diagnostics:  # reads unlabelled ground truth; returns numbers only, no RNG draws
            diagnostics["pseudo_labels"] = pseudo_label_quality(state, train)
        lr = optimizer.param_groups[0]["lr"]
        steps, losses = train_one_epoch(  # phase B
            model, fourier, loader, state, optimizer, cfg,
            device=device, max_steps=cfg.loop.max_steps_per_epoch, generator=generator, epoch=epoch,
        )
        scheduler.step()
        stats = EpochStats(epoch, steps, losses, lr, state.num_levels, time.perf_counter() - start, diagnostics)
        result.history.append(stats)
        result.state = state
        log.info("epoch %d: %d steps, lr %.5f, losses %s", epoch, steps, lr, losses)

        last = epoch == num_epochs - 1
        if last or (cfg.loop.eval_every and (epoch + 1) % cfg.loop.eval_every == 0):
            scores = evaluate_transductive(
                model.backbone, datasets.train_eval, cfg.eval,
                batch_size=cfg.optim.batch_size, num_workers=cfg.loop.num_workers, device=device, generator=generator,
            )
            result.evaluations.append((epoch, scores))
            log.info("epoch %d: Train ACC Unlabelled %s", epoch, scores[cfg.eval.metric])
        if checkpoint_dir is not None:
            args = (model, optimizer, scheduler, epoch, cfg, provenance)
            save_checkpoint(Path(checkpoint_dir) / "last.pt", *args)
            keep = cfg.loop.keep_checkpoint_every
            if keep and (epoch + 1) % keep == 0:
                save_checkpoint(Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt", *args)
    return result
