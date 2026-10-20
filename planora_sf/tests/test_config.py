from pathlib import Path

import pytest

from planora_sf.config import (
    CUTOFF_TABLE,
    DataConfig,
    FourierConfig,
    LatentConfig,
    LoopConfig,
    PlanoraSFConfig,
    SelExConfig,
)

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_defaults_match_spec_reference_values():
    cfg = PlanoraSFConfig()
    assert cfg.selex_compat is True
    assert cfg.selex.lam == 0.35
    assert (cfg.weights.low, cfg.weights.high, cfg.weights.cls) == (0.1, 0.1, 0.1)
    assert cfg.latent.dim == 65536
    assert cfg.backbone.grad_from_block == 9
    assert (cfg.selex.tau_unsup, cfg.selex.tau_sup) == (1.0, 0.07)
    assert cfg.fourier.snr_target_db == 15.0
    assert cfg.fourier.blur_kernel_size == 5
    assert (cfg.optim.lr, cfg.optim.momentum, cfg.optim.epochs) == (0.1, 0.9, 200)


def test_table7_cutoffs_and_for_dataset():
    assert {name: table[15] for name, table in CUTOFF_TABLE.items()} == {
        "cub": 23, "aircraft": 27, "scars": 27, "pets": 18, "herbarium_19": 13,
    }
    cub = PlanoraSFConfig.for_dataset("cub")
    assert cub.fourier.t_low() == cub.fourier.t_high() == 23
    assert cub.selex.smoothing_alpha == 1.0 and cub.selex.balanced
    herb = PlanoraSFConfig.for_dataset("herbarium_19", snr_db=20)
    assert herb.fourier.t_low() == 32 and not herb.selex.balanced
    with pytest.raises(KeyError):
        PlanoraSFConfig.for_dataset("sku_unknown")


def test_unset_cutoff_raises_and_high_override():
    with pytest.raises(ValueError, match="cut-off T is not set"):
        FourierConfig().t_low()
    cfg = FourierConfig(cutoff=10, cutoff_high=12)
    assert (cfg.t_low(), cfg.t_high()) == (10, 12)


def test_target_mode_follows_selex_compat():
    assert PlanoraSFConfig(selex_compat=True).resolved_target_mode == "compat"
    assert PlanoraSFConfig(selex_compat=False).resolved_target_mode == "paper"
    explicit = PlanoraSFConfig(selex_compat=True, selex=SelExConfig(target_mode="radius"))
    assert explicit.resolved_target_mode == "radius"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LatentConfig(high_fraction_mode="bogus"),
        lambda: LatentConfig(dim=7),
        lambda: FourierConfig(blur_kernel_size=4),
        lambda: SelExConfig(lam=1.5),
        lambda: PlanoraSFConfig(base="other"),
        lambda: DataConfig(labelled_fraction=0.0),
        lambda: LoopConfig(prototype_features="other"),
        lambda: LoopConfig(max_steps_per_epoch=0),
    ],
)
def test_invalid_values_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_yaml_template_equals_dataclass_defaults():
    assert PlanoraSFConfig.from_yaml(DEFAULT_YAML) == PlanoraSFConfig()


def test_dict_round_trip_and_unknown_keys():
    cfg = PlanoraSFConfig.for_dataset("pets")
    assert PlanoraSFConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(KeyError):
        PlanoraSFConfig.from_dict({"fourier": {"not_a_field": 1}})
