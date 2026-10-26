"""GCD evaluator API (spec §7).

``predict`` clusters ℓ2-normalised CLS features of the whole training set with
semi-supervised k-means (BSSK, or plain SS-k-means; U-26). Cluster ``j <
n_known`` is anchored to known class ``known_class_ids[j]``, because labelled
samples are fixed to their class. Every sample is therefore either ``KNOWN``
(with a class id) or in a ``NOVEL_CLUSTER`` (with a cluster id only).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from planora_sf.config import EvalConfig
from planora_sf.eval.diagnostics import cluster_structure, novel_accuracy_by_tier, per_class_accuracy_v2, routing_matrix
from planora_sf.eval.metrics import AccuracyReport, split_cluster_acc_v1, split_cluster_acc_v2
from planora_sf.pseudo_labels.bssk import semi_supervised_kmeans


class AssignmentKind(str, Enum):
    KNOWN = "KNOWN"
    NOVEL_CLUSTER = "NOVEL_CLUSTER"


@dataclass
class GCDPrediction:
    cluster_ids: torch.Tensor  # [N] cluster index per sample
    is_known: torch.Tensor  # [N] bool: assigned to a known-class cluster
    known_class_ids: torch.Tensor  # [N] original known class id, -1 for novel clusters
    num_known_clusters: int

    @property
    def kinds(self) -> list[AssignmentKind]:
        return [AssignmentKind.KNOWN if k else AssignmentKind.NOVEL_CLUSTER for k in self.is_known.tolist()]


class GCDEvaluator:
    def __init__(self, num_known_classes: int, num_total_classes: int, cfg: Optional[EvalConfig] = None):
        if not 0 <= num_known_classes <= num_total_classes:
            raise ValueError("need 0 <= num_known_classes <= num_total_classes")
        self.num_known = num_known_classes
        self.num_total = num_total_classes
        self.cfg = cfg or EvalConfig()

    def predict(
        self,
        features: torch.Tensor,
        labelled_mask: torch.Tensor,
        targets: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> GCDPrediction:
        """Cluster all training features; labelled samples carry their known-class ``targets``."""
        feats = F.normalize(features.float(), dim=-1)
        cluster_size = math.ceil(feats.shape[0] / self.num_total) if self.cfg.clustering == "bssk" else None
        result = semi_supervised_kmeans(
            feats, labelled_mask, targets, self.num_total,
            cluster_size=cluster_size, max_iter=self.cfg.max_iter, n_init=self.cfg.n_init, generator=generator,
        )
        n_known_clusters = result.num_known_clusters
        is_known = result.labels < n_known_clusters
        class_ids = torch.full_like(result.labels, -1)
        class_ids[is_known] = result.known_class_ids[result.labels[is_known]]
        return GCDPrediction(result.labels, is_known, class_ids, n_known_clusters)

    def score(
        self, prediction: GCDPrediction, y_true: torch.Tensor, eval_mask: torch.Tensor
    ) -> dict[str, AccuracyReport]:
        """All/Known/Novel accuracy on ``eval_mask`` samples (the unlabelled split).

        Classes ``0 .. num_known - 1`` are the known classes, as in the SelEx target transform.
        """
        mask = eval_mask.bool()
        y = y_true[mask].cpu().numpy()
        pred = prediction.cluster_ids[mask].cpu().numpy()
        known = y < self.num_known
        return {"v1": split_cluster_acc_v1(y, pred, known), "v2": split_cluster_acc_v2(y, pred, known)}

    def score_detailed(
        self,
        prediction: GCDPrediction,
        y_true: torch.Tensor,
        eval_mask: torch.Tensor,
        tiers: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> dict:
        """``score()`` plus the diagnostics of ``eval.diagnostics`` (plan T1), as plain Python types.

        ``tiers`` maps a tier name to target ids (e.g. the SSB novel difficulty tiers).
        ``v1``/``v2`` are exactly ``score()``'s numbers, so they keep its NaN for an empty
        subset; ``cli.evaluate`` writes those as JSON ``null``.
        """
        mask = eval_mask.bool()
        y = y_true[mask].cpu().numpy()
        pred = prediction.cluster_ids[mask].cpu().numpy()
        n_clusters = prediction.num_known_clusters
        return {
            **{name: asdict(report) for name, report in self.score(prediction, y_true, eval_mask).items()},
            "routing": routing_matrix(y, pred, self.num_known, n_clusters),
            "per_class_v2": per_class_accuracy_v2(y, pred),
            "structure": cluster_structure(y, pred, self.num_known, n_clusters),
            "tiers": novel_accuracy_by_tier(y, pred, tiers) if tiers is not None else None,
        }
