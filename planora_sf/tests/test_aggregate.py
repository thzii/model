"""Decision-gate aggregator (master plan Task #6) on synthetic summary.json / cli.evaluate files,
plus one round trip through cli.train. Numbers are made up to land on each gate branch."""
import json

import pytest

from planora_sf.cli import aggregate, train

COMMIT, SPLIT = "c" * 40, "s" * 64


def _summary(tmp, label, seed, all_, known, novel, *, dataset="scars", backbone="dinov1_vitb16", commit=COMMIT,
             split=SPLIT, history=None, row="E0"):
    v = {"all": all_ / 100, "known": known / 100, "novel": novel / 100}
    data = {
        "label": label, "dataset": dataset, "backbone": backbone, "components": {"row": row},
        "provenance": {"seed": seed, "commit": commit, "commit_dirty": False, "split_sha256": split,
                       "config_sha256": "x" * 64},
        "final": {"epoch": 199, "v1": v, "v2": v}, "history": history or [], "epochs_run": 200,
    }
    path = tmp / "runs" / f"{label}_{dataset}_{backbone}_s{seed}" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    return path


def _eval(tmp, name, alls, novels, *, sha="a" * 64, frozen=False, crop=None, clustering="bssk", n_init=10,
          logged=None, pl_gap=None):
    per_seed = []
    for seed, (a, n) in enumerate(zip(alls, novels)):
        v = {"all": a / 100, "known": (a + 10) / 100, "novel": n / 100}
        per_seed.append({"seed": seed, "v1": v, "v2": v, "routing": {"absorption": 0.2, "leakage": 0.1},
                         "tiers": None})
    mean = lambda xs: sum(xs) / len(xs) / 100  # noqa: E731
    std = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 / 100  # noqa: E731
    data = {
        "checkpoint": {"path": None if frozen else f"/runs/{sha[:4]}/last.pt", "epoch": None if frozen else 199,
                       "sha256": None if frozen else sha, "frozen_backbone": frozen},
        "dataset": "scars", "backbone": "dinov1_vitb16", "split_sha256": SPLIT, "crop": crop,
        "evaluator": {"clustering": clustering, "n_init": n_init, "max_iter": 10},
        "per_seed": per_seed,
        "summary": {"v2.all": {"mean": mean(alls), "std": std(alls)}, "v2.novel": {"mean": mean(novels),
                                                                                   "std": std(novels)}},
        "logged": None if logged is None else {"linked": logged},
        "pseudo_labels": None if pl_gap is None else {"novel_minus_evaluator_novel": pl_gap,
                                                      "flag_novel_below_evaluator": pl_gap < -0.05},
    }
    path = tmp / "evals" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _run(capsys, *args):
    rc = aggregate.main([str(a) for a in args])
    out = capsys.readouterr()
    return rc, out


def _report(tmp_path, capsys, *args):
    rc, _ = _run(capsys, *args, "--out-json", tmp_path / "gates.json", "--out-md", tmp_path / "gates.md")
    assert rc == 0
    return json.loads((tmp_path / "gates.json").read_text())


@pytest.mark.parametrize("e0, e4p, outcome", [
    ((57.0, 76.0, 48.0), (61.0, 79.0, 53.0), "both pass"),
    ((53.0, 76.0, 43.0), (61.0, 79.0, 53.0), "grey zone"),
    ((52.0, 70.0, 43.0), (61.0, 79.0, 53.0), "E0 fails"),
    ((57.0, 76.0, 48.0), (58.0, 79.0, 47.0), "E4p fails"),
])
def test_gate2_branches(tmp_path, capsys, e0, e4p, outcome):
    _summary(tmp_path, "S2-E0", 42, *e0)
    _summary(tmp_path, "S2-E4p", 42, *e4p, row="E4")
    report = _report(tmp_path, capsys, "--runs", tmp_path / "runs")
    gate = report["gate2"]["scars"]
    assert outcome in gate["outcome"]
    if e0[1] == 70.0:
        assert gate["S2-E0"]["flags"] == ["known off by -6.9"]
    assert "## Gate 2" in (tmp_path / "gates.md").read_text(encoding="utf-8")


def test_gate2_cub_applies_the_gap_condition(tmp_path, capsys):
    _summary(tmp_path, "S2-E0", 42, 75.0, 80.0, 72.5, dataset="cub")  # All in band, Known - Novel = 7.5 > 3
    report = _report(tmp_path, capsys, "--runs", tmp_path / "runs")
    assert report["gate2"]["cub"]["S2-E0"]["status"].startswith("fail (Known - Novel")


def test_refusals(tmp_path, capsys):
    _summary(tmp_path, "S2-E0", 42, 57.0, 76.0, 48.0)
    _summary(tmp_path, "S2-E4p", 42, 61.0, 79.0, 53.0, commit="d" * 40)
    rc, out = _run(capsys, "--runs", tmp_path / "runs")
    assert rc == 2 and "2 commits" in out.err and "gate1" not in json.loads(out.out)
    rc, out = _run(capsys, "--runs", tmp_path / "runs", "--allow-mixed-commits")
    assert rc == 0 and "Warning: runs come from 2 commits" in out.out
    _summary(tmp_path, "S2-E0", 43, 57.0, 76.0, 48.0, split="t" * 64)
    rc, out = _run(capsys, "--runs", tmp_path / "runs", "--allow-mixed-commits")
    assert rc == 2 and "2 different split hashes" in out.err
    with pytest.raises(SystemExit):
        aggregate.main([])


def _grad_history(cos, base_norm, fourier_norm):
    return [{"epoch": e, "diagnostics": {"grad": {"mean": {"cos/base~fourier": cos, "norm/base": base_norm,
                                                           "norm/fourier": fourier_norm,
                                                           "block9/cos/base~fourier": 0.9}}}}
            for e in range(0, 200, 50)]


def test_gate4_screen_confirm_additivity_and_gradients(tmp_path, capsys):
    e4 = {42: 62.0, 43: 62.5, 44: 63.0}
    for seed, a in e4.items():
        _summary(tmp_path, "S2-E4p", seed, a, 80.0, 54.0, row="E4")
        _summary(tmp_path, "S4-F", seed, a + 2.0, 80.0, 57.0, row=None,
                 history=_grad_history(-0.3, 100.0, 1.0))
        _summary(tmp_path, "S4-D", seed, a + 0.5, 77.0, 56.0, row=None)
        _summary(tmp_path, "S4-FD", seed, a + 3.0, 80.0, 58.0, row=None)
    for seed, a in {42: 80.5, 43: 81.0, 44: 80.2}.items():
        _summary(tmp_path, "S4-v2-E0", seed, a, 90.0, 76.0, backbone="dinov2_vitb14_reg")
    report = _report(tmp_path, capsys, "--runs", tmp_path / "runs", "--sigma-eval", "0.3")
    g4 = report["gate4"]
    assert g4["delta_eval_pct"] == pytest.approx(1.0)  # max(1, 2 x 0.3)
    res = g4["by_dataset"]["scars"]
    assert res["S4-F"]["screen_pass"] is True and res["S4-F"]["confirmation"]["confirmed"] is True
    assert res["S4-D"]["screen_pass"] is False  # +0.5 All and Known -3
    add = res["additivity"]
    assert add["best_single"] == "S4-F" and add["confirmation"]["confirmed"] is True  # +1.0 on every seed
    assert add["interaction_mean"] == pytest.approx(0.5) and add["no_cancellation"] is True
    flags = res["S4-F"]["gradients"]["flags"]
    assert any("conflict base~fourier" in f for f in flags) and any("base dominates" in f for f in flags)
    assert res["S4-D"]["gradients"]["note"].startswith("no gradient monitor data")
    v2 = g4["dinov2"]["scars"]
    assert v2["provisional_pass"] is True and v2["above_80_on_3_seeds"] is True


def test_gate1_and_gate3_from_evaluations(tmp_path, capsys):
    # trained checkpoint: stable evaluator, linked; frozen floor; both with crop pairs that improve
    _eval(tmp_path, "e0_full", [57.0, 57.4, 56.8], [47.0, 47.5, 46.9], logged=True, pl_gap=-0.08)
    _eval(tmp_path, "e0_sskm", [56.0, 56.2, 55.9], [45.0, 45.2, 45.1], clustering="ss_kmeans")
    _eval(tmp_path, "e0_crop", [59.5, 59.9, 59.2], [50.0, 50.6, 49.9], crop={"margin": 0.1, "shape": "square"})
    _eval(tmp_path, "frozen_full", [40.0, 40.5, 40.2], [30.0, 30.4, 30.1], frozen=True)
    _eval(tmp_path, "frozen_crop", [43.0, 43.8, 43.1], [33.0, 33.9, 33.0], frozen=True,
          crop={"margin": 0.1, "shape": "square"})
    # second trained checkpoint whose log does not match
    _eval(tmp_path, "debias_full", [61.5, 61.7, 61.4], [53.0, 53.5, 53.1], sha="b" * 64, logged=False)
    report = _report(tmp_path, capsys, "--evals", tmp_path / "evals")
    g1 = report["gate1"]
    by_sha = {c["weights"]: c for c in g1["checkpoints"]}
    e0 = by_sha["sha256:" + "a" * 64]
    assert e0["decision"] == "usable" and e0["linked"] is True and e0["pl_flag"] is True
    assert e0["novel_spread_pct"] == pytest.approx(47.133 - 45.1, abs=1e-2)
    assert by_sha["sha256:" + "b" * 64]["decision"].startswith("exclude")
    assert by_sha["frozen:dinov1_vitb16"]["decision"] == "reference floor"
    assert g1["sigma_eval_pct"] == pytest.approx(0.3055, abs=1e-3)  # sd of 57.0, 57.4, 56.8 (largest trained)
    assert g1["outcome"].startswith("proceed to Step 2")
    g3 = report["gate3"]
    assert len(g3["pairs"]) == 2 and all(p["v2.all"]["verdict"] == "supported increase" for p in g3["pairs"])
    assert g3["outcome"].startswith("background bias supported")


def test_gate1_flags_an_evaluator_limited_checkpoint(tmp_path, capsys):
    _eval(tmp_path, "full", [57.0, 57.4, 56.8], [47.0, 47.5, 46.9])
    _eval(tmp_path, "long", [60.0, 60.2, 60.1], [53.0, 53.2, 53.1], n_init=100)  # Novel +6 with 100 inits
    report = _report(tmp_path, capsys, "--evals", tmp_path / "evals")
    assert report["gate1"]["checkpoints"][0]["decision"].startswith("evaluator-limited")
    assert report["gate1"]["outcome"].startswith("fix the evaluator first")
    assert report["gate3"]["outcome"].startswith("missing")


def test_round_trip_with_cli_train(tmp_path, capsys):
    for seed in (1, 2):
        assert train.main(["--dataset", "dummy", "--epochs", "1", "--max-steps", "1", "--batch-size", "16",
                           "--device", "cpu", "--num-workers", "0", "--seed", str(seed), "--label", "smoke",
                           "--checkpoint-dir", str(tmp_path / "runs" / f"s{seed}")]) == 0
    capsys.readouterr()
    report = _report(tmp_path, capsys, "--runs", tmp_path / "runs")
    (group,) = report["groups"]
    assert (group["label"], group["dataset"], group["backbone"], group["seeds"]) == ("smoke", "dummy", "tiny", [1, 2])
    assert report["gate2"] == {} and report["checks"]["errors"] == []


def test_console_output_survives_a_cp1252_terminal(monkeypatch):
    import io

    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr("sys.stdout", stream)
    aggregate._console("σ_eval = 0.3; Δ All → pass; Known − Novel ± 1; ≥ 80")
    stream.flush()
    assert buffer.getvalue().decode("cp1252").strip() == "sigma_eval = 0.3; d All -> pass; Known - Novel +/- 1; >= 80"
