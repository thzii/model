"""Aggregate runs and evaluations, pair them by seed and apply Decision Gates 1-4 (master plan Part 3, Task #6).

Inputs (files or directories, searched recursively):
  --runs    summary.json files written by cli.train --checkpoint-dir (Gates 2 and 4)
  --evals   cli.evaluate JSON files (Gate 1: σ_eval, evaluator sensitivity, log link, PL-acc; Gate 3: crop pairs)
  --audits  cli.audit_checkpoint JSON files (Gate 1: dataset identity)

Runs are grouped by (label, dataset, backbone); ``cli.train --label`` names them. The gate roles
default to the master plan run IDs: S2-E0, S2-E4p, S4-F, S4-D, S4-FD, S4-v2-E0, S4-v2-E4.
Aggregation is refused (exit 2) when runs come from different commits (``--allow-mixed-commits``
overrides), when one dataset has two split hashes, or when a (group, seed) pair appears twice.
All accuracies are reported in percentage points.

    python -m planora_sf.cli.aggregate --runs runs/ --evals runs/eval/ --audits audit/ \
        --out-json runs/gates.json --out-md runs/gates.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from planora_sf.cli.common import json_safe
from planora_sf.cli.compare_evals import IncomparableError, compare, paired_stats, weights_identity
from planora_sf.engine.grad_monitor import ADDITIVE_PARTS

MIN_DELTA_PCT = 1.0
# Gate 2 targets (v2 All/Known/Novel, DINOv1). Cars: master plan Gate 2; CUB: research plan G1 / R5.
TARGETS: dict[str, dict[str, dict[str, Any]]] = {
    "scars": {
        "E0": {"all": 56.9, "known": 76.9, "novel": 47.3, "gap_max": None, "source": "SelEx† (FourEx Table 1)"},
        "E4p": {"all": 62.9, "known": 80.0, "novel": 54.7, "gap_max": None, "source": "Hyp-SelEx (HypCD Table 1)"},
    },
    "cub": {
        "E0": {"all": 76.4, "known": 72.4, "novel": 78.4, "gap_max": 3.0, "source": "SelEx† (FourEx Table 4)"},
        "E4p": {"all": 79.8, "known": 75.8, "novel": 81.8, "gap_max": None, "source": "Hyp-SelEx (HypCD)"},
    },
}
BAND_PCT, GREY_PCT, FLAG_PCT = 3.0, 1.5, 5.0  # Gate 2: pass within ±3, grey up to 1.5 beyond, flag Known/Novel off by >5
SENSITIVITY_PCT = 5.0  # Gate 1: Novel moving more than 5 points across evaluator settings
SCREEN_ALL_PCT, SCREEN_KNOWN_DROP_PCT = 1.0, 2.0  # Gate 4 screening
CONFLICT_COS, DOMINANCE_RATIO = -0.1, 10.0  # Gate 4 gradient flags
DINOV2_PASS_PCT, DINOV2_TARGET_PCT = 79.0, 80.0


# ---------------------------------------------------------------- loading

def _json_files(paths: Iterable[Path], name: Optional[str] = None) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(path.rglob(name or "*.json"))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else 100.0 * x


def load_runs(paths: Sequence[Path]) -> tuple[list[dict], list[str]]:
    runs, warnings = [], []
    for path in _json_files(paths, "summary.json"):
        data = _load(path)
        final = data.get("final") or (data.get("evaluations") or [None])[-1]
        if "provenance" not in data or not final:
            warnings.append(f"{path}: no provenance or no final evaluation (pre-Task #2b summary?); skipped")
            continue
        prov = data["provenance"]
        runs.append({
            "path": str(path), "label": data.get("label") or data["components"].get("row") or "custom",
            "dataset": data.get("dataset"), "backbone": data.get("backbone"), "row": data["components"].get("row"),
            "seed": prov.get("seed"), "commit": prov.get("commit"), "dirty": prov.get("commit_dirty"),
            "split_sha256": prov.get("split_sha256"), "config_sha256": prov.get("config_sha256"),
            "v2": {k: _pct(final["v2"][k]) for k in ("all", "known", "novel")},
            "v1": {k: _pct(final["v1"][k]) for k in ("all", "known", "novel")},
            "history": data.get("history", []), "epochs_run": data.get("epochs_run", len(data.get("history", []))),
        })
    return runs, warnings


def load_evals(paths: Sequence[Path]) -> list[dict]:
    out = []
    for path in _json_files(paths):
        data = _load(path)
        if isinstance(data, Mapping) and "per_seed" in data and "evaluator" in data and "checkpoint" in data:
            out.append({**data, "_path": str(path)})
    return out


def load_audits(paths: Sequence[Path]) -> dict[str, dict]:
    """Checkpoint path -> audit report (cli.audit_checkpoint output)."""
    out = {}
    for path in _json_files(paths):
        data = _load(path)
        for report in data.get("checkpoints", []) if isinstance(data, Mapping) else []:
            out[str(Path(report["path"]).resolve())] = report
    return out


# ---------------------------------------------------------------- checks and groups

def consistency_checks(runs: Sequence[dict], allow_mixed_commits: bool) -> dict[str, Any]:
    errors, warnings = [], []
    commits = sorted({str(r["commit"]) for r in runs})
    if len(commits) > 1:
        (warnings if allow_mixed_commits else errors).append(f"runs come from {len(commits)} commits: {commits}")
    dirty = [r["path"] for r in runs if r["dirty"]]
    if dirty:
        warnings.append(f"{len(dirty)} runs were trained from a modified tree: {dirty}")
    splits: dict[str, set] = defaultdict(set)
    for r in runs:
        splits[r["dataset"]].add(r["split_sha256"])
    for dataset, hashes in splits.items():
        if len(hashes) > 1:
            errors.append(f"dataset {dataset}: {len(hashes)} different split hashes")
    seen: dict[tuple, str] = {}
    for r in runs:
        key = (r["label"], r["dataset"], r["backbone"], r["seed"])
        if key in seen:
            errors.append(f"duplicate run {key}: {seen[key]} and {r['path']}")
        seen[key] = r["path"]
    return {"commits": commits, "errors": errors, "warnings": warnings}


def group_runs(runs: Sequence[dict]) -> dict[tuple, dict[int, dict]]:
    groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in runs:
        groups[(r["label"], r["dataset"], r["backbone"])][r["seed"]] = r
    return dict(groups)


def _ms(values: Sequence[Optional[float]]) -> dict[str, Optional[float]]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return {"mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}


def describe_group(by_seed: Mapping[int, dict]) -> dict[str, Any]:
    rs = [by_seed[s] for s in sorted(by_seed)]
    out = {"seeds": sorted(by_seed), "n": len(rs)}
    for k in ("all", "known", "novel"):
        out[f"v2.{k}"] = _ms([r["v2"][k] for r in rs])
        out[f"v1.{k}"] = _ms([r["v1"][k] for r in rs])
    out["gap"] = _ms([None if r["v2"]["known"] is None or r["v2"]["novel"] is None
                      else r["v2"]["known"] - r["v2"]["novel"] for r in rs])
    return out


def _find(groups, label: str, dataset: Optional[str] = None, backbone: Optional[str] = None) -> dict[tuple, dict]:
    return {key: g for key, g in groups.items() if key[0] == label and (dataset is None or key[1] == dataset)
            and (backbone is None or key[2] == backbone)}


# ---------------------------------------------------------------- Gate 1

def _default_eval(evals: Sequence[dict]) -> Optional[dict]:
    plain = [e for e in evals if not e.get("crop")]
    standard = [e for e in plain if (e["evaluator"]["clustering"], e["evaluator"]["n_init"], e["evaluator"]["max_iter"])
                == ("bssk", 10, 10)]
    return (standard or plain or [None])[0]


def gate1(evals: Sequence[dict], audits: Mapping[str, dict], sigma_override: Optional[float]) -> dict[str, Any]:
    by_weights: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by_weights[weights_identity(e)].append(e)
    checkpoints, sigmas = [], []
    for identity, items in sorted(by_weights.items()):
        base = _default_eval(items)
        sigma = None if base is None else _pct(base["summary"]["v2.all"]["std"])
        trained = not items[0]["checkpoint"].get("frozen_backbone")
        if sigma is not None and trained:
            sigmas.append(sigma)
        settings = {}
        for e in items:
            if e.get("crop"):
                continue
            ev = e["evaluator"]
            settings[f"{ev['clustering']} {ev['n_init']}x{ev['max_iter']}"] = _pct(e["summary"]["v2.novel"]["mean"])
        novel = [v for v in settings.values() if v is not None]
        spread = max(novel) - min(novel) if len(novel) > 1 else None
        links = [e["logged"] for e in items if e.get("logged")]
        linked = None if not links else all(link["linked"] for link in links)
        pl = [e["pseudo_labels"] for e in items if e.get("pseudo_labels")]
        path = items[0]["checkpoint"].get("path")
        audit = audits.get(str(Path(path).resolve())) if path else None
        if linked is False:
            decision = "exclude: rescored v2 does not match the logged run"
        elif spread is not None and spread > SENSITIVITY_PCT:
            decision = "evaluator-limited: fix the evaluator, re-score, then continue"
        else:
            decision = "usable" if trained else "reference floor"
        checkpoints.append({
            "weights": identity, "path": path, "dataset": items[0]["dataset"], "backbone": items[0].get("backbone"),
            "trained": trained, "audit_dataset": None if audit is None else audit.get("dataset"),
            "v2_all_pct": None if base is None else _pct(base["summary"]["v2.all"]["mean"]),
            "v2_novel_pct": None if base is None else _pct(base["summary"]["v2.novel"]["mean"]),
            "sigma_eval_pct": sigma, "novel_by_evaluator_pct": settings, "novel_spread_pct": spread,
            "linked": linked, "pl_flag": None if not pl else pl[0]["flag_novel_below_evaluator"],
            "pl_novel_minus_evaluator_pct": None if not pl else _pct(pl[0]["novel_minus_evaluator_novel"]),
            "decision": decision,
        })
    sigma = sigma_override if sigma_override is not None else (max(sigmas) if sigmas else None)
    delta = max(MIN_DELTA_PCT, 2.0 * sigma) if sigma is not None else MIN_DELTA_PCT
    pairs = []
    usable = [c for c in checkpoints if c["trained"] and c["decision"] == "usable" and c["v2_all_pct"] is not None]
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            if a["dataset"] == b["dataset"]:
                diff = a["v2_all_pct"] - b["v2_all_pct"]
                pairs.append({"a": a["path"], "b": b["path"], "delta_all_pct": diff,
                              "verdict": "difference detected" if abs(diff) >= delta else "no difference detected"})
    if not checkpoints:
        outcome = "missing: no cli.evaluate outputs"
    elif any(c["decision"].startswith("evaluator-limited") for c in checkpoints if c["trained"]):
        outcome = "fix the evaluator first (cheap), re-score, then run Step 2"
    else:
        outcome = "proceed to Step 2 on Stanford Cars (S2-E0, S2-E4p); excluded checkpoints are not evidence"
    return {"checkpoints": checkpoints, "sigma_eval_pct": sigma,
            "sigma_source": "--sigma-eval" if sigma_override is not None else ("max over trained checkpoints"
                                                                                 if sigmas else None),
            "delta_eval_pct": delta, "pairs": pairs, "outcome": outcome}


# ---------------------------------------------------------------- Gate 2

def _role_status(desc: Optional[dict], target: Mapping[str, Any]) -> dict[str, Any]:
    if desc is None or desc["n"] == 0:
        return {"status": "missing"}
    mean = {k: desc[f"v2.{k}"]["mean"] for k in ("all", "known", "novel")}
    d = mean["all"] - target["all"]
    if abs(d) <= BAND_PCT:
        status = "pass"
    elif abs(d) <= BAND_PCT + GREY_PCT and desc["n"] < 3:
        status = "grey: run 2 more seeds and decide on the mean"
    else:
        status = "fail"
    gap = desc["gap"]["mean"]
    if status == "pass" and target["gap_max"] is not None and gap is not None and gap > target["gap_max"]:
        status = f"fail (Known - Novel {gap:.1f} > {target['gap_max']})"
    flags = [f"{k} off by {mean[k] - target[k]:+.1f}" for k in ("known", "novel")
             if mean[k] is not None and abs(mean[k] - target[k]) > FLAG_PCT]
    return {"status": status, "n": desc["n"], "mean_pct": mean, "delta_all_pct": d, "target": dict(target),
            "flags": flags}


def gate2(groups, roles: Mapping[str, str]) -> dict[str, Any]:
    out = {}
    for dataset, targets in TARGETS.items():
        e0 = _find(groups, roles["e0"], dataset, "dinov1_vitb16")
        e4p = _find(groups, roles["e4p"], dataset, "dinov1_vitb16")
        if not e0 and not e4p:
            continue
        r0 = _role_status(describe_group(next(iter(e0.values()))) if e0 else None, targets["E0"])
        r4 = _role_status(describe_group(next(iter(e4p.values()))) if e4p else None, targets["E4p"])
        if r0["status"] == "missing":
            outcome = "missing S2-E0"
        elif r0["status"].startswith("grey") or r4["status"].startswith("grey"):
            outcome = "grey zone: run 2 more seeds, then decide on the mean"
        elif r0["status"] != "pass":
            outcome = "E0 fails: the base pipeline deviates; stop Step 3-4 training and return to R1 diagnosis"
        elif r4["status"] == "missing":
            outcome = "E0 passes; S2-E4p missing"
        elif r4["status"] != "pass":
            outcome = "E0 passes, E4p fails: the HypCD port deviates (hyp_* monitors, parity tests, H-08/H-09)"
        else:
            outcome = "both pass: pipeline reproduced; Δ(E4p - E0) is informational (blocks and D differ)"
        out[dataset] = {"S2-E0": r0, "S2-E4p": r4, "outcome": outcome}
    return out


# ---------------------------------------------------------------- Gate 3

def gate3(evals: Sequence[dict]) -> dict[str, Any]:
    by_weights: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by_weights[weights_identity(e)].append(e)
    pairs = []
    for identity, items in sorted(by_weights.items()):
        full = [e for e in items if not e.get("crop")]
        for cropped in (e for e in items if e.get("crop")):
            match = next((f for f in full if f["evaluator"]["clustering"] == cropped["evaluator"]["clustering"]
                          and f["evaluator"]["n_init"] == cropped["evaluator"]["n_init"]
                          and f["evaluator"]["max_iter"] == cropped["evaluator"]["max_iter"]), None)
            if match is None:
                continue
            try:
                result = compare(match, cropped)
            except IncomparableError as err:
                pairs.append({"weights": identity, "error": str(err)})
                continue
            pairs.append({"weights": identity, "frozen": bool(cropped["checkpoint"].get("frozen_backbone")),
                          "crop": cropped["crop"], "v2.all": result["metrics"]["v2.all"],
                          "v2.novel": result["metrics"]["v2.novel"]})
    valid = [p for p in pairs if "error" not in p]
    frozen = [p for p in valid if p["frozen"]]
    trained = [p for p in valid if not p["frozen"]]
    inc = lambda p: p["v2.all"]["verdict"] == "supported increase"  # noqa: E731
    flat = lambda p: p["v2.all"]["verdict"].startswith("not detected")  # noqa: E731
    if not frozen or not trained:
        outcome = "missing: need crop/full pairs for the frozen backbone and a trained checkpoint (S2-E0)"
    elif all(inc(p) for p in valid):
        outcome = "background bias supported: run E0-crop (1 seed, recalibrated T) as a separate protocol"
    elif all(inc(p) for p in frozen) and all((p["v2.all"]["mean"] or 0.0) <= 0 for p in trained):
        outcome = ("inconclusive: distribution shift dominates on the trained checkpoint; "
                   "retrain with crops only for domain reasons")
    elif all(flat(p) for p in valid):
        outcome = "background is not a significant bottleneck: no crop retraining for accuracy"
    else:
        outcome = "mixed evidence: report the pairs; no retraining decision"
    return {"pairs": pairs, "outcome": outcome,
            "note": "for PLANORA production (detector SKU crops) the crop protocol has separate domain value"}


# ---------------------------------------------------------------- Gate 4

def paired(a: Mapping[int, dict], b: Mapping[int, dict], key: str = "all") -> dict[str, Any]:
    """Deltas a - b (v2, points) over the seeds both groups have."""
    seeds = sorted(set(a) & set(b))
    deltas = [a[s]["v2"][key] - b[s]["v2"][key] for s in seeds]
    return {"seeds": seeds, **paired_stats(deltas)}


def confirm(stats: Mapping[str, Any], delta_eval: float) -> dict[str, Any]:
    """Research plan §4.4 plus master plan Gate 4: n >= 3, all positive, mean >= δ, mean - 2 SE > 0."""
    if stats["n"] < 3:
        return {"confirmed": None, "reason": f"needs 3 paired seeds, has {stats['n']}"}
    ok = stats["all_positive"] and stats["mean"] >= delta_eval and stats["mean"] - 2 * stats["se"] > 0
    return {"confirmed": bool(ok), "reason": f"all > 0: {stats['all_positive']}, mean {stats['mean']:+.2f} "
                                             f"vs δ {delta_eval:.2f}, mean - 2SE {stats['mean'] - 2 * stats['se']:+.2f}"}


def gradient_flags(group: Mapping[int, dict]) -> dict[str, Any]:
    """Median over the second half of training of each run's per-epoch gradient means (Gate 4 diagnostics)."""
    flags, medians = [], {}
    for seed, run in sorted(group.items()):
        half = run["epochs_run"] / 2
        epochs = [h["diagnostics"]["grad"]["mean"] for h in run["history"]
                  if h["epoch"] >= half and "grad" in h.get("diagnostics", {})]
        if not epochs:
            continue
        keys = {k for e in epochs for k in e if k.startswith(("cos/", "norm/"))}  # all blocks, not per block
        med = {k: statistics.median(e[k] for e in epochs if k in e) for k in sorted(keys)}
        medians[seed] = med
        for k, v in med.items():
            pair = k.split("/", 1)[1]
            if k.startswith("cos/") and not pair.startswith("base~hyp_") and v < CONFLICT_COS:
                flags.append(f"seed {seed}: conflict {pair} (median cos {v:+.2f})")
        norms = {p: med[f"norm/{p}"] for p in ADDITIVE_PARTS if f"norm/{p}" in med}
        if len(norms) > 1:
            top = max(norms, key=norms.get)
            rest = [v for p, v in norms.items() if p != top]
            if all(norms[top] > DOMINANCE_RATIO * v for v in rest):
                flags.append(f"seed {seed}: {top} dominates (norm {norms[top]:.3g} > {DOMINANCE_RATIO:g}x the others)")
    return {"medians_second_half": medians, "flags": flags,
            "note": None if medians else "no gradient monitor data (cli.train --grad-monitor-every)"}


def gate4(groups, roles: Mapping[str, str], delta_eval: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in sorted({k[1] for k in groups}):
        ref = _find(groups, roles["e4p"], dataset, "dinov1_vitb16")
        cands = {name: _find(groups, roles[name], dataset, "dinov1_vitb16") for name in ("e4f", "e4d", "trio")}
        if not ref or not any(cands.values()):
            continue
        e4 = next(iter(ref.values()))
        result: dict[str, Any] = {}
        for name, found in cands.items():
            if not found:
                continue
            g = next(iter(found.values()))
            d_all, d_known = paired(g, e4, "all"), paired(g, e4, "known")
            screen = None
            if d_all["n"]:
                screen = d_all["mean"] >= SCREEN_ALL_PCT and d_known["mean"] >= -SCREEN_KNOWN_DROP_PCT
            result[roles[name]] = {"vs": roles["e4p"], "delta_all": d_all, "delta_known_mean": d_known["mean"],
                                   "screen_pass": screen, "confirmation": confirm(d_all, delta_eval),
                                   "gradients": gradient_flags(g)}
        if all(cands.values()):
            f, d, fd = (next(iter(cands[n].values())) for n in ("e4f", "e4d", "trio"))
            singles = {roles["e4f"]: f, roles["e4d"]: d}
            best = max(singles, key=lambda n: paired(singles[n], e4)["mean"] or -math.inf)
            vs_best = paired(fd, singles[best])
            seeds = sorted(set(e4) & set(f) & set(d) & set(fd))
            inter = [(fd[s]["v2"]["all"] - e4[s]["v2"]["all"]) - ((f[s]["v2"]["all"] - e4[s]["v2"]["all"])
                                                                + (d[s]["v2"]["all"] - e4[s]["v2"]["all"]))
                     for s in seeds]
            mean_i = float(np.mean(inter)) if inter else None
            result["additivity"] = {
                "best_single": best, "trio_vs_best_single": vs_best, "confirmation": confirm(vs_best, delta_eval),
                "interaction_per_seed": inter, "interaction_mean": mean_i,
                "no_cancellation": None if mean_i is None else mean_i >= -delta_eval,
            }
        out[dataset] = result
    dinov2 = {}
    for dataset in sorted({k[1] for k in groups}):
        found = _find(groups, roles["v2_e0"], dataset, "dinov2_vitb14_reg")
        if not found:
            continue
        g = next(iter(found.values()))
        desc = describe_group(g)
        alls = [g[s]["v2"]["all"] for s in sorted(g)]
        e4 = _find(groups, roles["v2_e4"], dataset, "dinov2_vitb14_reg")
        dinov2[dataset] = {
            "n": desc["n"], "v2_all_mean": desc["v2.all"]["mean"],
            "provisional_pass": desc["v2.all"]["mean"] >= DINOV2_PASS_PCT,  # master plan Gate 4: >= 79
            "above_80_on_3_seeds": len(alls) >= 3 and all(a >= DINOV2_TARGET_PCT for a in alls),
            "e4": describe_group(next(iter(e4.values()))) if e4 else None,
        }
    return {"delta_eval_pct": delta_eval, "by_dataset": out, "dinov2": dinov2}


# ---------------------------------------------------------------- report

_ASCII = str.maketrans({"σ": "sigma", "δ": "delta", "Δ": "d", "−": "-", "–": "-", "→": "->", "±": "+/-",
                        "≥": ">=", "†": "+"})


def _console(text: str) -> None:
    """Print ``text``, transliterated to ASCII when the console encoding cannot show it (Windows cp1252)."""
    try:
        text.encode(getattr(sys.stdout, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        text = text.translate(_ASCII)
    print(text)


def _f(x: Optional[float], digits: int = 2) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Decision gates", ""]
    checks = report["checks"]
    lines += [f"- Commits: {', '.join(checks['commits']) or '–'}"]
    lines += [f"- ERROR: {e}" for e in checks["errors"]] + [f"- Warning: {w}" for w in checks["warnings"]]
    lines += ["", "## Runs", "", "| Label | Dataset | Backbone | Seeds | v2 All | Known | Novel | Known − Novel |",
              "|---|---|---|---|---|---|---|---|"]
    for g in report["groups"]:
        lines.append(f"| {g['label']} | {g['dataset']} | {g['backbone']} | {','.join(map(str, g['seeds']))} | "
                     f"{_f(g['v2.all']['mean'])} ± {_f(g['v2.all']['sd'])} | {_f(g['v2.known']['mean'])} | "
                     f"{_f(g['v2.novel']['mean'])} | {_f(g['gap']['mean'])} |")
    g1 = report["gate1"]
    lines += ["", "## Gate 1: checkpoint audit", "", f"σ_eval = {_f(g1['sigma_eval_pct'])} pt "
              f"({g1['sigma_source'] or 'not measured'}); δ_eval = {_f(g1['delta_eval_pct'])} pt", "",
              "| Weights | Dataset (audit) | v2 All | Novel | σ_eval | Novel spread | Linked | PL − eval Novel "
              "| Decision |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in g1["checkpoints"]:
        lines.append(f"| {c['path'] or c['weights']} | {c['dataset']} ({c['audit_dataset'] or '–'}) | "
                     f"{_f(c['v2_all_pct'])} | {_f(c['v2_novel_pct'])} | {_f(c['sigma_eval_pct'])} | "
                     f"{_f(c['novel_spread_pct'])} | {c['linked']} | {_f(c['pl_novel_minus_evaluator_pct'])} | "
                     f"{c['decision']} |")
    lines += [f"- {p['a']} vs {p['b']}: Δ All {p['delta_all_pct']:+.2f} pt, {p['verdict']}" for p in g1["pairs"]]
    lines += ["", f"**Outcome:** {g1['outcome']}", "", "## Gate 2: paper reproduction", ""]
    if not report["gate2"]:
        lines.append("No runs labelled as the S2-E0 / S2-E4p roles on CUB or Stanford Cars (DINOv1).")
    for dataset, g in report["gate2"].items():
        for role in ("S2-E0", "S2-E4p"):
            r = g[role]
            if r["status"] == "missing":
                lines.append(f"- {dataset} {role}: missing")
                continue
            lines.append(f"- {dataset} {role}: v2 All {_f(r['mean_pct']['all'])} (target {r['target']['all']}, "
                         f"Δ {r['delta_all_pct']:+.2f}, n = {r['n']}) → **{r['status']}**"
                         + (f"; flags: {'; '.join(r['flags'])}" if r["flags"] else ""))
        lines.append(f"- **{dataset} outcome:** {g['outcome']}")
    g3 = report["gate3"]
    lines += ["", "## Gate 3: bounding-box crop", ""]
    for p in g3["pairs"]:
        if "error" in p:
            lines.append(f"- {p['weights']}: {p['error']}")
            continue
        a = p["v2.all"]
        lines.append(f"- {'frozen' if p['frozen'] else 'trained'} {p['weights'][:23]}: Δ All {_f(a['mean'])} "
                     f"(sd {_f(a['sd'])}, threshold {_f(a['threshold'])}) → {a['verdict']}; "
                     f"Δ Novel {_f(p['v2.novel']['mean'])}")
    lines += ["", f"**Outcome:** {g3['outcome']} ({g3['note']})", "", "## Gate 4: combined losses and DINOv2", "",
              f"δ_eval = {_f(report['gate4']['delta_eval_pct'])} pt", ""]
    if not report["gate4"]["by_dataset"] and not report["gate4"]["dinov2"]:
        lines.append("No S4-* runs paired with an S2-E4p reference, and no DINOv2 E0 runs.")
    for dataset, res in report["gate4"]["by_dataset"].items():
        for name, r in res.items():
            if name == "additivity":
                c = r["confirmation"]
                lines.append(f"- {dataset} trio vs {r['best_single']}: Δ {_f(r['trio_vs_best_single']['mean'])}, "
                             f"confirmed {c['confirmed']} ({c['reason']}); interaction {_f(r['interaction_mean'])}, "
                             f"no cancellation {r['no_cancellation']}")
                continue
            c = r["confirmation"]
            lines.append(f"- {dataset} {name} vs {r['vs']}: Δ All {_f(r['delta_all']['mean'])} over seeds "
                         f"{r['delta_all']['seeds']}, screen {r['screen_pass']}, confirmed {c['confirmed']} "
                         f"({c['reason']})" + (f"; gradient flags: {'; '.join(r['gradients']['flags'])}"
                                               if r["gradients"]["flags"] else ""))
    for dataset, r in report["gate4"]["dinov2"].items():
        lines.append(f"- {dataset} DINOv2 E0: v2 All {_f(r['v2_all_mean'])} (n = {r['n']}): provisional pass "
                     f"{r['provisional_pass']}, ≥ 80 on 3 seeds {r['above_80_on_3_seeds']}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="*", default=[], help="summary.json files or directories")
    parser.add_argument("--evals", type=Path, nargs="*", default=[], help="cli.evaluate JSON files or directories")
    parser.add_argument("--audits", type=Path, nargs="*", default=[], help="cli.audit_checkpoint JSON files")
    parser.add_argument("--sigma-eval", type=float, default=None, help="sigma_eval in points (default: from --evals)")
    parser.add_argument("--allow-mixed-commits", action="store_true")
    for flag, default in (("e0", "S2-E0"), ("e4p", "S2-E4p"), ("e4f", "S4-F"), ("e4d", "S4-D"), ("trio", "S4-FD"),
                          ("v2-e0", "S4-v2-E0"), ("v2-e4", "S4-v2-E4")):
        parser.add_argument(f"--{flag}", default=default, help=f"run label of this gate role (default {default})")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.runs or args.evals):
        build_parser().error("give --runs and/or --evals")
    runs, warnings = load_runs(args.runs)
    evals = load_evals(args.evals)
    audits = load_audits(args.audits)
    checks = consistency_checks(runs, args.allow_mixed_commits)
    checks["warnings"] = warnings + checks["warnings"]
    groups = group_runs(runs)
    roles = {"e0": args.e0, "e4p": args.e4p, "e4f": args.e4f, "e4d": args.e4d, "trio": args.trio,
             "v2_e0": args.v2_e0, "v2_e4": args.v2_e4}
    report: dict[str, Any] = {
        "checks": checks,
        "groups": [{"label": k[0], "dataset": k[1], "backbone": k[2], **describe_group(g)}
                   for k, g in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0])))],
    }
    if checks["errors"]:  # runs that cannot be paired get no gate verdicts
        print("refusing to aggregate:\n  " + "\n  ".join(checks["errors"]), file=sys.stderr)
        print(json.dumps(json_safe(report), indent=2, allow_nan=False))
        return 2
    g1 = gate1(evals, audits, args.sigma_eval)
    report.update(gate1=g1, gate2=gate2(groups, roles), gate3=gate3(evals),
                  gate4=gate4(groups, roles, g1["delta_eval_pct"]))
    text = json.dumps(json_safe(report), indent=2, allow_nan=False)
    markdown = render_markdown(report)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
    _console(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
