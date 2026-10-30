"""Audit checkpoints and logged metrics: which dataset, which components, which known-sample share w.

Master plan Part 1.1. Nothing is trained or evaluated; checkpoints are only read.

Each checkpoint yields independent dataset votes from tensor shapes (number of known classes,
number of classes) and, for native ``planora_sf`` checkpoints, from the stored config (dataset,
Fourier cut-off). Shape votes work even when ``config.dataset`` is null and for foreign
checkpoints (official HypCD / DebGCD / SimGCD code), which carry no config.

Logged metrics give w = (All - Novel) / (Known - Novel), the share of GT-known samples in the
scored unlabelled set. It holds for v1 and v2 alike and depends only on dataset + split, so it
is compared with each dataset's reference w as an interval that includes rounding of the logs.

    python -m planora_sf.cli.audit_checkpoint runs/fourex/last.pt \\
        --logged v1=62.39,77.72,54.72 --logged v2=60.10,74.05,53.12 [--scars-root /data/scars]

Checkpoints are loaded with ``torch.load(weights_only=True)``: tensors and plain containers only,
no code execution. ``--unsafe-load`` lifts that, for checkpoints you produced yourself.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

# Reference class counts and known-sample shares (SSB splits, 50% of known-class images labelled).
DATASETS: dict[str, dict[str, Any]] = {
    "cub": {
        "num_classes": 200, "num_known": 100, "fourier_cutoff": 23,
        # CUB-200-2011 train has 29-30 images per class (5,994 = 194 x 30 + 6 x 29), so the 100 SSB known
        # classes hold 2,994-3,000 images; w = (n_k - n_k // 2) / (5,994 - n_k // 2).
        "w_range": (0.3329, 0.3338), "w_source": "analytic: 29-30 train images per class",
    },
    "scars": {
        "num_classes": 196, "num_known": 98, "fourier_cutoff": 27,
        # Only GCD Table 1's rounded |D_L| ~ 2.0K, |D_U| ~ 6.1K is known without the data; pass
        # --scars-root for the exact value.
        "w_range": (0.3145, 0.3364), "w_source": "loose: GCD Table 1 rounding (2.0K labelled); use --scars-root",
    },
}
DUMMY_CUTOFF = 6  # cli.train's synthetic config


# ---------------------------------------------------------------- loading

class AuditError(RuntimeError):
    pass


def load(path: Path, unsafe: bool = False) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=not unsafe)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure with a hint
        hint = "" if unsafe else " (if you produced this file yourself, retry with --unsafe-load)"
        raise AuditError(f"cannot load {path}: {type(exc).__name__}: {str(exc).splitlines()[0]}{hint}") from exc


def _state_dict(obj: Any) -> tuple[str, Mapping[str, torch.Tensor], Optional[Mapping[str, Any]]]:
    """(format, state dict, raw native checkpoint or None)."""
    if not isinstance(obj, Mapping):
        raise AuditError(f"unsupported checkpoint object: {type(obj).__name__}")
    if "model" in obj and isinstance(obj["model"], Mapping) and "config" in obj:
        return "native", obj["model"], obj
    for wrapper in ("state_dict", "model", "teacher", "student"):
        inner = obj.get(wrapper)
        if isinstance(inner, Mapping) and any(isinstance(v, torch.Tensor) for v in inner.values()):
            return f"foreign ({wrapper})", inner, None
    if any(isinstance(v, torch.Tensor) for v in obj.values()):
        return "foreign (state_dict)", obj, None
    raise AuditError("no tensors found: not a model checkpoint")


# ---------------------------------------------------------------- fingerprints

def _weight(sd: Mapping[str, torch.Tensor], prefix: str) -> Optional[torch.Tensor]:
    """Direction weight of a (weight-normalised) linear layer, in either torch weight-norm format."""
    for suffix in (".parametrizations.weight.original1", ".weight_v", ".weight"):
        tensor = sd.get(prefix + suffix)
        if tensor is not None and tensor.ndim == 2:
            return tensor
    return None


def backbone_fingerprint(sd: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = list(sd)
    block_ids = {int(m.group(1)) for k in keys if (m := re.search(r"(?:^|\.)blocks\.(\d+)\.", k))}
    patch = next((sd[k] for k in keys if re.search(r"patch_embed(\.proj)?\.weight$", k) and sd[k].ndim == 4), None)
    cls_token = next((sd[k] for k in keys if k.endswith("cls_token")), None)
    registers = next((sd[k] for k in keys if k.endswith("register_tokens")), None)
    info = {
        "num_blocks": len(block_ids) or None,
        "embed_dim": int(cls_token.shape[-1]) if cls_token is not None else None,
        "patch_size": int(patch.shape[-1]) if patch is not None else None,
        "register_tokens": int(registers.shape[1]) if registers is not None else 0,
    }
    family = None
    if info["embed_dim"] == 768 and info["num_blocks"] == 12:
        family = {16: "dinov1_vitb16", 14: "dinov2_vitb14" + ("_reg" if info["register_tokens"] else "")}.get(
            info["patch_size"])
    info["family"] = family or ("tiny/other" if info["embed_dim"] else None)
    return info


# (state-dict prefix, what its output dimension counts). Native names first, then the official code's.
_HEADS = [
    ("cls_head.fc", "num_known"),  # FourEx L_cls head (planora_sf)
    ("debgcd.debiased.last_layer", "num_classes"),  # DebGCD ψ_s (planora_sf)
    ("debgcd.detector.last_layer", "two_num_known"),  # DebGCD OVA heads, 2M outputs (planora_sf)
    ("head.last_layer", "latent_dim"),  # DINOHead (planora_sf)
    ("1.last_layer_deb", "num_classes"),  # official DebGCD projector (nn.Sequential(backbone, projector))
    ("1.last_layer_ood", "two_num_known"),
    ("1.last_layer", "num_classes_or_latent"),  # SimGCD/DebGCD prototype classifier, or a projection head
    ("0.last_layer", "latent_dim"),  # official HypCD projection head file (DINOHead, ToPoincare)
    ("last_layer", "num_classes_or_latent"),
]


def head_fingerprint(sd: Mapping[str, torch.Tensor]) -> dict[str, int]:
    out: dict[str, int] = {}
    for prefix, meaning in _HEADS:
        weight = _weight(sd, prefix)
        if weight is not None and prefix not in out:
            out[prefix] = int(weight.shape[0])
            out[f"{prefix}:meaning"] = meaning  # type: ignore[assignment]
    return out


def _shape_votes(heads: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(dataset, evidence) for every head output that matches a reference class count."""
    votes = []
    for prefix, meaning in _HEADS:
        if prefix not in heads:
            continue
        n = heads[prefix]
        for name, ref in DATASETS.items():
            expected = {"num_known": [ref["num_known"]], "num_classes": [ref["num_classes"]],
                        "two_num_known": [2 * ref["num_known"]],
                        "num_classes_or_latent": [ref["num_classes"]]}.get(meaning, [])
            if n in expected:
                votes.append((name, f"{prefix} has {n} outputs ({meaning})"))
    return votes


def _config_votes(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    votes = []
    dataset = config.get("dataset")
    if dataset in DATASETS:
        votes.append((dataset, f"config.dataset = {dataset!r}"))
    cutoff = (config.get("fourier") or {}).get("cutoff")
    for name, ref in DATASETS.items():
        if cutoff == ref["fourier_cutoff"]:
            votes.append((name, f"config.fourier.cutoff = {cutoff} (FX-A Table 7)"))
    if cutoff == DUMMY_CUTOFF and (config.get("backbone") or {}).get("name") == "tiny":
        votes.append(("dummy", "tiny backbone with the synthetic smoke config"))
    return votes


def _components(config: Mapping[str, Any]) -> dict[str, Any]:
    from planora_sf.config import PlanoraSFConfig

    try:
        cfg = PlanoraSFConfig.from_dict(config)
    except Exception as exc:  # noqa: BLE001 - written by an older/newer code version
        return {"error": f"config not readable by this code version: {exc}"}
    return {
        **cfg.components(),
        "backbone": cfg.backbone.name,
        "trainable_blocks": f"{cfg.backbone.grad_from_block}..",
        "latent_dim": cfg.latent.dim,
        "smoothing_alpha": cfg.selex.smoothing_alpha,
        "target_mode": cfg.resolved_target_mode,
        "reproduce_label_order_bug": cfg.selex.reproduce_label_order_bug,
        "epochs_scheduled": cfg.optim.epochs,
        "eval": {"clustering": cfg.eval.clustering, "n_init": cfg.eval.n_init, "max_iter": cfg.eval.max_iter},
    }


def verdict(votes: Sequence[tuple[str, str]]) -> str:
    names = {name for name, _ in votes}
    if not names:
        return "unknown"
    return names.pop() if len(names) == 1 else "conflict: " + " vs ".join(sorted(names))


def audit_checkpoint(path: Path, *, unsafe: bool = False) -> dict[str, Any]:
    fmt, sd, native = _state_dict(load(path, unsafe))
    heads = head_fingerprint(sd)
    votes = _shape_votes(heads)
    report: dict[str, Any] = {
        "path": str(path),
        "format": fmt,
        "num_tensors": sum(isinstance(v, torch.Tensor) for v in sd.values()),
        "backbone": backbone_fingerprint(sd),
        "heads": {k: v for k, v in heads.items() if not k.endswith(":meaning")},
    }
    if native is not None:
        config = native["config"] if isinstance(native["config"], Mapping) else {}
        votes += _config_votes(config)
        report["epoch"] = native.get("epoch")
        report["components"] = _components(config)
        report["has_optimizer_state"] = "optimizer" in native
        stored = native.get("provenance") if isinstance(native.get("provenance"), Mapping) else native
        keys = ("seed", "commit", "commit_dirty", "config_sha256", "split_sha256", "hub")
        report["provenance"] = {k: stored.get(k) for k in keys}  # all None before Task #2b checkpoints
        report["rescore"] = "cli.evaluate --checkpoint (native checkpoint)"
    else:
        report["rescore"] = "not directly: foreign weights need a key mapping (master plan Task #1/2a)"
    report["dataset_votes"] = [{"dataset": d, "evidence": e} for d, e in votes]
    report["dataset"] = verdict(votes)
    return report


# ---------------------------------------------------------------- w from logged metrics

def parse_logged(text: str) -> tuple[str, list[str]]:
    """``"v2=60.10,74.05,53.12"`` -> ("v2", ["60.10", "74.05", "53.12"]); the label is optional."""
    label, _, values = text.rpartition("=")
    parts = [p.strip() for p in values.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected [label=]All,Known,Novel, got {text!r}")
    for p in parts:
        float(p)
    return label or "logged", parts


def w_interval(all_: str, known: str, novel: str) -> dict[str, Any]:
    """w = (All - Novel)/(Known - Novel), with the interval implied by rounding each logged value.

    Each value is taken as ±half a unit of its last printed decimal; w is monotone in each
    argument, so the extremes are at the corners of that box.
    """
    values = [float(v) for v in (all_, known, novel)]
    halves = [0.5 * 10.0 ** Decimal(v).as_tuple().exponent for v in (all_, known, novel)]
    a, k, n = values
    if k == n:
        raise ValueError("Known equals Novel: w is undefined")
    corners = []
    for da, dk, dn in itertools.product(*[(-h, h) for h in halves]):
        if (k + dk) != (n + dn):
            corners.append((a + da - (n + dn)) / (k + dk - (n + dn)))
    w = (a - n) / (k - n)
    return {"w": round(w, 5), "w_min": round(min(corners), 5), "w_max": round(max(corners), 5),
            "consistent_ordering": min(k, n) <= a <= max(k, n)}


def _overlaps(lo: float, hi: float, ref: Sequence[float]) -> bool:
    return lo <= ref[1] and ref[0] <= hi


def exact_reference_w(name: str, root: str) -> dict[str, Any]:
    """Exact w from the data: the split summary that training and evaluation build."""
    from planora_sf.config import DataConfig
    from planora_sf.data.factory import build_sources, build_split

    train_source, _ = build_sources(name, root)
    summary = build_split(name, train_source, DataConfig()).summary(train_source.targets)
    w = summary["unlabelled_known_images"] / summary["unlabelled_images"]
    return {"w": round(w, 5), "split": summary}


def audit_logged(logged: Sequence[tuple[str, list[str]]], references: Mapping[str, Mapping[str, Any]]) -> dict:
    rows = []
    for label, values in logged:
        row = {"label": label, "all_known_novel": [float(v) for v in values], **w_interval(*values)}
        row["compatible_datasets"] = [
            name for name, ref in references.items()
            if _overlaps(row["w_min"], row["w_max"], ref.get("w_exact_range") or ref["w_range"])
        ]
        rows.append(row)
    same_set = all(_overlaps(r["w_min"], r["w_max"], (rows[0]["w_min"], rows[0]["w_max"])) for r in rows)
    return {"rows": rows, "all_rows_share_one_eval_set": same_set if len(rows) > 1 else None}


# ---------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="native last.pt or foreign state dicts")
    parser.add_argument("--logged", action="append", type=parse_logged, default=[], metavar="[LABEL=]ALL,KNOWN,NOVEL",
                        help="logged accuracies of one run (repeat for v1 and v2), in %% as printed")
    parser.add_argument("--cub-root", default=None, help="CUB root, for the exact CUB w")
    parser.add_argument("--scars-root", default=None, help="Stanford Cars root, for the exact Cars w")
    parser.add_argument("--unsafe-load", action="store_true", help="torch.load(weights_only=False): own files only")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.checkpoints and not args.logged:
        parser.error("give at least one checkpoint or one --logged triple")

    references = {name: dict(ref) for name, ref in DATASETS.items()}
    for name, root in (("cub", args.cub_root), ("scars", args.scars_root)):
        if root:
            exact = exact_reference_w(name, root)
            references[name]["w_exact"] = exact["w"]
            references[name]["w_exact_range"] = (exact["w"] - 5e-5, exact["w"] + 5e-5)
            references[name]["split"] = exact["split"]

    report: dict[str, Any] = {"references": references, "checkpoints": []}
    failed = False
    for path in args.checkpoints:
        try:
            report["checkpoints"].append(audit_checkpoint(path, unsafe=args.unsafe_load))
        except AuditError as exc:
            report["checkpoints"].append({"path": str(path), "error": str(exc)})
            failed = True
    if args.logged:
        report["logged"] = audit_logged(args.logged, references)
        datasets = {c.get("dataset") for c in report["checkpoints"] if "dataset" in c}
        if len(args.checkpoints) == 1 and len(datasets) == 1:
            ds = datasets.pop()
            report["logged"]["matches_checkpoint_dataset"] = (
                all(ds in r["compatible_datasets"] for r in report["logged"]["rows"]) if ds in DATASETS else None
            )

    text = json.dumps(report, indent=2, default=list)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    _print_summary(report)
    return 1 if failed else 0


def _print_summary(report: Mapping[str, Any]) -> None:
    lines = ["", "== audit summary =="]
    for c in report["checkpoints"]:
        if "error" in c:
            lines.append(f"{c['path']}: ERROR {c['error']}")
            continue
        comp = c.get("components", {})
        row = comp.get("row") if isinstance(comp, Mapping) else None
        lines.append(f"{c['path']}: dataset={c['dataset']}  format={c['format']}  epoch={c.get('epoch')}  "
                     f"backbone={c['backbone']['family']}  row={row}")
    for r in report.get("logged", {}).get("rows", []):
        lines.append(f"{r['label']}: w={r['w']:.4f} [{r['w_min']:.4f}, {r['w_max']:.4f}]  "
                     f"compatible with {r['compatible_datasets'] or 'no reference dataset'}")
    logged = report.get("logged", {})
    if logged.get("all_rows_share_one_eval_set") is False:
        lines.append("WARNING: the logged rows imply different evaluation sets")
    if logged.get("matches_checkpoint_dataset") is False:
        lines.append("WARNING: the logged w does not fit the checkpoint's dataset")
    print("\n".join(lines), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
