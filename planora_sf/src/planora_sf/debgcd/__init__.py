"""DebGCD (ICLR 2025) ported onto SelEx in the adapted form of plan §2.2 (experiments E2, E3)."""
from planora_sf.debgcd.heads import DebGCDHeads, DebiasedPrototypeHead, SemanticDistributionDetector
from planora_sf.debgcd.losses import (
    DebGCDTerms,
    adl_labelled,
    adl_unlabelled,
    certainty,
    consistency,
    debgcd_terms,
    distribution_score,
    sdl_labelled,
    sdl_unlabelled,
)
from planora_sf.debgcd.surrogate import align_novel_clusters, centroid_posterior, permute_level

__all__ = [
    "DebGCDHeads",
    "DebGCDTerms",
    "DebiasedPrototypeHead",
    "SemanticDistributionDetector",
    "adl_labelled",
    "adl_unlabelled",
    "align_novel_clusters",
    "centroid_posterior",
    "certainty",
    "consistency",
    "debgcd_terms",
    "distribution_score",
    "permute_level",
    "sdl_labelled",
    "sdl_unlabelled",
]
