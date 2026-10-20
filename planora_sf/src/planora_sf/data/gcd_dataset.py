"""Torch datasets over an image source and a GCD split."""
from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from planora_sf.data.sources import ImageSource
from planora_sf.data.splits import GCDSplit


class GCDTrainDataset(Dataset):
    """Training split with labelled samples first, then unlabelled (SelEx MergedDataset order).

    Items are ``(image or views, target, is_labelled, index)``. ``target`` is the
    true mapped class for every sample; it is needed to score transductive
    evaluation. The training loop must only use ``train_targets``, which hides
    unlabelled targets as -1. ``index`` is the position in this dataset and keys
    the pseudo-label state.
    """

    def __init__(self, source: ImageSource, split: GCDSplit, transform: Optional[Callable] = None):
        self.source = source
        self.split = split
        self.transform = transform
        self.source_indices = np.concatenate([split.labelled_indices, split.unlabelled_indices])
        self.num_labelled = int(split.labelled_indices.size)
        self.targets = torch.as_tensor(split.map_targets(source.targets[self.source_indices]))
        self.is_labelled = torch.arange(len(self.source_indices)) < self.num_labelled

    @property
    def num_unlabelled(self) -> int:
        return len(self) - self.num_labelled

    @property
    def train_targets(self) -> torch.Tensor:
        """Targets visible to training: the class for labelled samples, -1 for unlabelled."""
        return torch.where(self.is_labelled, self.targets, torch.full_like(self.targets, -1))

    def with_transform(self, transform: Optional[Callable]) -> "GCDTrainDataset":
        clone = copy.copy(self)
        clone.transform = transform
        return clone

    def __len__(self) -> int:
        return int(self.source_indices.size)

    def __getitem__(self, index: int):
        image = self.source.load(int(self.source_indices[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index]), bool(self.is_labelled[index]), index


class GCDTestDataset(Dataset):
    """Disjoint test split with every class mapped by the training split's class order."""

    def __init__(self, source: ImageSource, split: GCDSplit, transform: Optional[Callable] = None):
        self.source = source
        self.transform = transform
        self.targets = torch.as_tensor(split.map_targets(source.targets))

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, index: int):
        image = self.source.load(index)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index]), index
