"""Optimiser and LR schedule from the SelEx code (ct.py:208-215; FourEx states none of these, U-23)."""
from __future__ import annotations

import torch
from torch import nn

from planora_sf.config import OptimConfig


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.SGD:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimConfig) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * cfg.eta_min_factor)
