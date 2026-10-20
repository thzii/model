"""Shared fixtures. Everything here is synthetic: random tensors only, no images, no checkpoints.

Set PLANORA_SF_TEST_SEED to rerun the suite under a different random seed.
"""
from __future__ import annotations

import os

import pytest
import torch

from helpers import make_tiny_config

SEED = int(os.environ.get("PLANORA_SF_TEST_SEED", "0"))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(SEED)


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(SEED)


@pytest.fixture
def tiny_cfg():
    return make_tiny_config()
