"""Configuration dataclasses for PLANORA-SF.

Every default carries its provenance from ``PLANORA_SF_Method_Spec.md``:
a paper equation or table (SX-P / SX-S / FX-P / FX-A), the official SelEx
code (``ct.py`` = ``methods/contrastive_training/contrastive_training.py``),
or a ``U-xx`` scaffold default. Scaffold defaults are engineering
placeholders, not paper values, and exist so they can be ablated.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

# FX-A Table 7: cut-off frequency per dataset at each SNR level (dB).
CUTOFF_TABLE: dict[str, dict[int, int]] = {
    "cub": {5: 2, 10: 8, 15: 23, 20: 45, 25: 55},
    "aircraft": {5: 2, 10: 9, 15: 27, 20: 50, 25: 55},
    "scars": {5: 3, 10: 11, 15: 27, 20: 47, 25: 55},
    "pets": {5: 2, 10: 7, 15: 18, 20: 38, 25: 55},
    "herbarium_19": {5: 1, 10: 2, 15: 13, 20: 32, 25: 50},
}

# SelEx README smoothing values (U-24). The SelEx paper instead says
# 0.5 for fine-grained and 0.1 for coarse-grained datasets.
README_SMOOTHING: dict[str, float] = {
    "cub": 1.0,
    "scars": 1.0,
    "pets": 0.5,
    "aircraft": 0.5,
    "cifar10": 0.1,
    "cifar100": 0.1,
    "imagenet_100": 0.1,
}

# SelEx README: long-tailed datasets run without the cluster-size cap (U-30).
UNBALANCED_DATASETS = frozenset({"herbarium_19"})


def _check_choice(name: str, value: Any, choices: set) -> None:
    if value not in choices:
        raise ValueError(f"{name}={value!r}; expected one of {sorted(choices)}")


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass
class BackboneConfig:
    name: str = "dinov1_vitb16"  # FX-P §5.1; "dinov2_vitb14_reg" or "tiny" (synthetic tests)
    grad_from_block: int = 9  # FX-P §5.1, FX-A §9.1: train blocks 9-11
    train_final_norm: bool = False  # SelEx code never unfreezes the final norm (ct.py:746-758)
    image_size: int = 224  # ct.py:738

    def __post_init__(self) -> None:
        _check_choice("backbone.name", self.name, {"dinov1_vitb16", "dinov2_vitb14_reg", "tiny"})
        if self.grad_from_block < 0:
            raise ValueError("backbone.grad_from_block must be >= 0")
        _check_positive("backbone.image_size", self.image_size)


@dataclass
class HeadConfig:
    # SelEx DINOHead (models/vision_transformer.py:264-298); output dim is LatentConfig.dim
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    num_layers: int = 3

    def __post_init__(self) -> None:
        _check_positive("head.hidden_dim", self.hidden_dim)
        _check_positive("head.bottleneck_dim", self.bottleneck_dim)
        _check_positive("head.num_layers", self.num_layers)


@dataclass
class LatentConfig:
    dim: int = 65536  # D, SelEx ct.py:741; FourEx does not state it (U-12)
    fraction_of_half: bool = True  # d is a fraction of D/2, not D (U-13)
    high_fraction_mode: str = "one_minus"  # d_h = (1 - f_h/T_h)·D/2; "ratio" = f_h/T_h (U-11)
    renormalize_slices: bool = False  # SelEx code does not re-normalise slices (U-14, D-14)

    def __post_init__(self) -> None:
        if self.dim < 2 or self.dim % 2:
            raise ValueError(f"latent.dim must be an even integer >= 2, got {self.dim}")
        _check_choice("latent.high_fraction_mode", self.high_fraction_mode, {"one_minus", "ratio"})


@dataclass
class FourierConfig:
    snr_target_db: float = 15.0  # FX-P §4
    cutoff: Optional[int] = None  # T; from fourier.snr calibration or CUTOFF_TABLE
    cutoff_high: Optional[int] = None  # None -> same as cutoff, i.e. T_l = T_h (U-08)
    blur_kernel_size: int = 5  # FX-P §5.1
    blur_sigma: float = 1.1  # U-03: OpenCV's default sigma for k = 5
    blur_anchor: bool = False  # U-03: blur only the FFT input
    strict_inequality: bool = True  # U-06: |u| < f, as printed
    band_pass: bool = False  # U-07: pure high-pass, as printed
    per_image_cutoffs: bool = True  # U-02
    low_cutoff_min: int = 1  # U-02: f_l in {1 .. T_l}
    high_cutoff_min: int = 0  # U-02: f_h in {0 .. T_h - 1}
    filtered_view: str = "x1"  # U-01: filter the first augmented view
    extra_views: bool = True  # U-01: l and h are extra forward passes
    snr_metric: str = "box"  # U-09: max(|k_u|, |k_v|)

    def __post_init__(self) -> None:
        if self.blur_kernel_size < 1 or self.blur_kernel_size % 2 == 0:
            raise ValueError("fourier.blur_kernel_size must be a positive odd integer")
        _check_positive("fourier.blur_sigma", self.blur_sigma)
        for name in ("cutoff", "cutoff_high"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"fourier.{name} must be >= 1, got {value}")
        if self.low_cutoff_min < 0 or self.high_cutoff_min < 0:
            raise ValueError("fourier cut-off minimums must be >= 0")
        _check_choice("fourier.filtered_view", self.filtered_view, {"x1", "x2"})
        _check_choice("fourier.snr_metric", self.snr_metric, {"box", "radial"})

    def t_low(self) -> int:
        if self.cutoff is None:
            raise ValueError(
                "Fourier cut-off T is not set: calibrate it with planora_sf.fourier.snr "
                "or use PlanoraSFConfig.for_dataset()"
            )
        return self.cutoff

    def t_high(self) -> int:
        return self.cutoff_high if self.cutoff_high is not None else self.t_low()


@dataclass
class SelExConfig:
    lam: float = 0.35  # SX-P §5.1
    smoothing_alpha: float = 0.5  # SX-P §5.1 (fine-grained); for_dataset() applies README values (U-24)
    tau_unsup: float = 1.0  # ct.py:668
    tau_sup: float = 0.07  # ct.py:59
    similarity: str = "neg_euclidean"  # ct.py:118,174 (D-06)
    use_loss: str = "soft_ce"  # D-01; "bce" is the literal SX-12 reading
    target_mode: Optional[str] = None  # None -> "compat" if selex_compat else "paper" (U-21)
    balanced: bool = True  # BSSK cluster-size cap; False for long-tailed data (U-30)
    kmeans_max_iter: int = 10  # ct.py:540-543
    kmeans_n_init: int = 1  # ct.py:540-543
    reproduce_label_order_bug: bool = False  # U-22 / D-08: aligned by default

    def __post_init__(self) -> None:
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("selex.lam must be in [0, 1]")
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("selex.smoothing_alpha must be in [0, 1]")
        _check_positive("selex.tau_unsup", self.tau_unsup)
        _check_positive("selex.tau_sup", self.tau_sup)
        _check_choice("selex.similarity", self.similarity, {"neg_euclidean", "dot"})
        _check_choice("selex.use_loss", self.use_loss, {"soft_ce", "bce"})
        if self.target_mode is not None:
            _check_choice("selex.target_mode", self.target_mode, {"compat", "paper", "radius"})
        _check_positive("selex.kmeans_max_iter", self.kmeans_max_iter)
        _check_positive("selex.kmeans_n_init", self.kmeans_n_init)


@dataclass
class LossWeights:
    # FX-P Table 5: the reported result uses 0.1 for all three
    low: float = 0.1
    high: float = 0.1
    cls: float = 0.1


@dataclass
class ClsHeadConfig:
    enabled: bool = True
    input: str = "cls"  # U-17: linear head on the backbone CLS token
    loss: str = "sigmoid_bce"  # U-17: one-hot sigmoid BCE; "softmax_ce" is the alternative
    num_classes: Optional[int] = None  # None -> number of known classes

    def __post_init__(self) -> None:
        _check_choice("cls_head.input", self.input, {"cls", "z"})
        _check_choice("cls_head.loss", self.loss, {"sigmoid_bce", "softmax_ce"})


@dataclass
class OptimConfig:
    # SelEx code (U-23: FourEx states none of these except the batch size)
    lr: float = 0.1  # ct.py:657
    momentum: float = 0.9  # ct.py:208
    weight_decay: float = 5e-5  # bash_scripts/contrastive_train.sh (argparse default is 1e-4)
    epochs: int = 200  # ct.py:662
    eta_min_factor: float = 1e-3  # ct.py:211-215: cosine down to lr * 1e-3
    batch_size: int = 128  # SX-P §5.1, FX-A §9.1


@dataclass
class EvalConfig:
    clustering: str = "bssk"  # SX-S §4.1; "ss_kmeans" is FX-A §9.1's wording (U-26)
    n_init: int = 10  # k_means.py:123
    max_iter: int = 10  # k_means.py:122
    metric: str = "v2"  # Train ACC Unlabelled_v2

    def __post_init__(self) -> None:
        _check_choice("eval.clustering", self.clustering, {"bssk", "ss_kmeans"})
        _check_choice("eval.metric", self.metric, {"v1", "v2"})
        _check_positive("eval.n_init", self.n_init)
        _check_positive("eval.max_iter", self.max_iter)


@dataclass
class DataConfig:
    known_class_fraction: float = 0.5  # SX-P §5.1 (CIFAR-100 would use 0.8)
    labelled_fraction: float = 0.5  # SX-P §5.1: 50% of known-class images are labelled
    split_seed: int = 0  # SelEx subsample_instances seeds numpy with 0
    stratified: bool = False  # SelEx/GCD sample globally over known-class images, not per class
    use_ssb_splits: bool = True  # CUB/Aircraft/Cars use the SSB known/novel class lists

    def __post_init__(self) -> None:
        for name in ("known_class_fraction", "labelled_fraction"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"data.{name} must be in (0, 1]")


@dataclass
class LoopConfig:
    num_workers: int = 4  # SelEx scripts
    prototype_features: str = "train_view0"  # D-11 parity; "test_transform" is the alternative
    eval_every: int = 0  # evaluate every n epochs; 0 = only after the last epoch
    max_steps_per_epoch: Optional[int] = None  # dry runs only; None = a full SelEx epoch
    crop_pct: float = 0.875  # ct.py:712

    def __post_init__(self) -> None:
        _check_choice("loop.prototype_features", self.prototype_features, {"train_view0", "test_transform"})
        if self.num_workers < 0 or self.eval_every < 0:
            raise ValueError("loop.num_workers and loop.eval_every must be >= 0")
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < 1:
            raise ValueError("loop.max_steps_per_epoch must be >= 1 or None")


@dataclass
class PlanoraSFConfig:
    selex_compat: bool = True  # spec §8: reproduce official SelEx code behaviour
    base: str = "selex"  # FX-A §8: "selex" (FourEx) or "gcd" (FourGCD)
    dataset: Optional[str] = None
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    fourier: FourierConfig = field(default_factory=FourierConfig)
    selex: SelExConfig = field(default_factory=SelExConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    cls_head: ClsHeadConfig = field(default_factory=ClsHeadConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)

    def __post_init__(self) -> None:
        _check_choice("base", self.base, {"selex", "gcd"})

    @property
    def resolved_target_mode(self) -> str:
        if self.selex.target_mode is not None:
            return self.selex.target_mode
        return "compat" if self.selex_compat else "paper"

    @classmethod
    def for_dataset(cls, name: str, snr_db: int = 15) -> "PlanoraSFConfig":
        """Reference configuration for a benchmark dataset (FX-A Table 7, SelEx README)."""
        if name not in CUTOFF_TABLE:
            raise KeyError(f"no reference cut-off for dataset {name!r}; calibrate T with fourier.snr")
        if snr_db not in CUTOFF_TABLE[name]:
            raise KeyError(f"no reference cut-off for SNR {snr_db} dB; choose from {sorted(CUTOFF_TABLE[name])}")
        cfg = cls(dataset=name)
        cfg.fourier.cutoff = CUTOFF_TABLE[name][snr_db]
        cfg.fourier.snr_target_db = float(snr_db)
        cfg.selex.smoothing_alpha = README_SMOOTHING.get(name, cfg.selex.smoothing_alpha)
        cfg.selex.balanced = name not in UNBALANCED_DATASETS
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanoraSFConfig":
        return _from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PlanoraSFConfig":
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


def _from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        factory = fields[name].default_factory
        if factory is not dataclasses.MISSING and dataclasses.is_dataclass(factory):
            if not isinstance(value, Mapping):
                raise TypeError(f"{cls.__name__}.{name} must be a mapping")
            value = _from_dict(factory, value)
        kwargs[name] = value
    return cls(**kwargs)
