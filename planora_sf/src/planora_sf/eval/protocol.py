"""Dataset-level feature extraction and the transductive GCD evaluation protocol (spec §7)."""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from planora_sf.config import EvalConfig
from planora_sf.data.gcd_dataset import GCDTrainDataset
from planora_sf.eval.evaluator import GCDEvaluator
from planora_sf.eval.metrics import AccuracyReport
from planora_sf.models.backbone import BackboneAdapter


@torch.no_grad()
def extract_dataset_features(
    backbone: BackboneAdapter,
    dataset: GCDTrainDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    device: Union[str, torch.device] = "cpu",
    view: int = 0,
) -> torch.Tensor:
    """ℓ2-normalised CLS features for every sample, in dataset index order (pseudo-labels rely on it).

    When the dataset yields several views, only ``view`` is used (SelEx uses view 0 of
    the training transform for prototypes; D-11).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    was_training = backbone.training
    backbone.eval()
    feats, expected = [], 0
    for images, _, _, index in loader:
        if index[0].item() != expected:
            raise RuntimeError("feature extraction must visit samples in index order")
        expected += index.numel()
        if isinstance(images, (list, tuple)):
            images = images[view]
        feats.append(F.normalize(backbone(images.to(device)), dim=-1).cpu())
    backbone.train(was_training)
    return torch.cat(feats)


def evaluate_transductive(
    backbone: BackboneAdapter,
    dataset: GCDTrainDataset,
    cfg: EvalConfig,
    *,
    batch_size: int,
    num_workers: int = 0,
    device: Union[str, torch.device] = "cpu",
    generator: Optional[torch.Generator] = None,
) -> dict[str, AccuracyReport]:
    """Cluster all training images (labelled ones fixed) and score the unlabelled ones (``Train ACC Unlabelled``).

    ``dataset`` should use the test transform, e.g. ``GCDDatasets.train_eval``.
    """
    feats = extract_dataset_features(
        backbone, dataset, batch_size=batch_size, num_workers=num_workers, device=device
    )
    evaluator = GCDEvaluator(dataset.split.num_known, dataset.split.num_classes, cfg)
    prediction = evaluator.predict(feats, dataset.is_labelled, dataset.train_targets, generator=generator)
    return evaluator.score(prediction, dataset.targets, ~dataset.is_labelled)
