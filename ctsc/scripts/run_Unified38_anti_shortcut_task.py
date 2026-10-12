'Run one Unified38 anti-shortcut boundary task on the branch.'

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import torch

from ctsc.precursor.public_source_catalog import SOURCE_GROUP_TO_PLATFORM, dsrl_catalog_records, infer_public_task_source_group


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ORDER = [
    "full",
    "current_only",
    "cost_step_only",
    "history_without_current_cost",
    "history_obs_action_step_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--profiles", type=str, nargs="+", default=PROFILE_ORDER[1:])
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--step-q-low", type=float, default=0.75)
    parser.add_argument("--step-q-high", type=float, default=0.90)
    parser.add_argument("--outcome-q-low", type=float, default=0.75)
    parser.add_argument("--outcome-q-high", type=float, default=0.90)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=str(cwd), check=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative_delta(lhs: float, rhs: float) -> float:
    return float(lhs - rhs)


def _boundary_tag(task_result: dict[str, Any]) -> str:
    full_step = float(task_result["full_reference"]["step_catastrophic_auprc"])
    current_step = float(task_result["profiles"]["current_only"]["step_catastrophic_auprc"])
    cost_step = float(task_result["profiles"]["cost_step_only"]["step_catastrophic_auprc"])
    history_step = float(task_result["profiles"]["history_without_current_cost"]["step_catastrophic_auprc"])
    hist_obs_step = float(task_result["profiles"]["history_obs_action_step_only"]["step_catastrophic_auprc"])

    full_minus_current = full_step - current_step
    history_minus_current = history_step - current_step
    history_minus_cost = history_step - cost_step
    hist_obs_minus_current = hist_obs_step - current_step

    if full_minus_current <= 0.02 and history_minus_current <= 0.01 and hist_obs_minus_current <= 0.01:
        return "boundary-weak / mostly-shortcut-like"
    if history_minus_current >= 0.035 and history_minus_cost >= 0.015 and hist_obs_minus_current >= 0.015:
        return "boundary-stronger-than-expected"
    if history_minus_current >= 0.02 and hist_obs_minus_current >= 0.005 and cost_step >= history_step - 0.03:
        return "non-cost-history-visible but cost-dominant"
    if full_minus_current >= 0.03:
        return "cost-heavy but not current-only"
    return "mixed-prefix-support"


def _interpretation_note(task_result: dict[str, Any]) -> str:
    full_step = float(task_result["full_reference"]["step_catastrophic_auprc"])
    current_step = float(task_result["profiles"]["current_only"]["step_catastrophic_auprc"])
    cost_step = float(task_result["profiles"]["cost_step_only"]["step_catastrophic_auprc"])
    history_step = float(task_result["profiles"]["history_without_current_cost"]["step_catastrophic_auprc"])
    hist_obs_step = float(task_result["profiles"]["history_obs_action_step_only"]["step_catastrophic_auprc"])

    parts: list[str] = []
    if full_step > current_step + 0.03:
        parts.append("full stays meaningfully above current_only")
    elif full_step > current_step + 0.01:
        parts.append("full stays modestly above current_only")
    else:
        parts.append("current_only remains close to full")

    if history_step > current_step + 0.02:
        parts.append("masked-history stays above current_only")
    elif history_step > current_step:
        parts.append("masked-history is only slightly above current_only")
    else:
        parts.append("masked-history does not stay above current_only")

    if history_step > cost_step + 0.01:
        parts.append("history_without_current_cost stays above cost_step_only")
    elif cost_step > history_step + 0.01:
        parts.append("cost_step_only stays above history_without_current_cost")
    else:
        parts.append("history_without_current_cost and cost_step_only stay close")

    if hist_obs_step > current_step + 0.01:
        parts.append("non-cost history remains visible after removing all cost")
    else:
        parts.append("history_obs_action_step_only remains limited")
    if hist_obs_step < history_step - 0.01:
        parts.append("historical cost remains important")
    return "; ".join(parts)


def _catalog_source_platform(task_name: str) -> str:
    catalog = {str(row["task_name"]): row for row in dsrl_catalog_records()}
    row = catalog.get(str(task_name))
    if row is not None:
        return str(row["source_platform"])
    source_group = infer_public_task_source_group(str(task_name))
    return SOURCE_GROUP_TO_PLATFORM.get(str(source_group), "unknown_source_platform")


def _profile_metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    label_step = metrics["label_step"]["test"]["overall"]
    label_outcome = metrics["label_outcome"]["test"]["overall"]
    return {
        "model_name": str(metrics["model_name"]),
        "model_family": str(metrics["model_family"]),
        "input_view": str(metrics["input_view"]),
        "feature_profile": str(metrics["feature_profile"]),
        "step_catastrophic_auprc": float(label_step["catastrophic_auprc"]),
        "outcome_catastrophic_auprc": float(label_outcome["catastrophic_auprc"]),
        "step_catastrophic_ece": float(label_step["catastrophic_ece"]),
        "step_lead_time_median": float(label_step["lead_time_median_at_0.5"]),
        "step_lead_time_mean": float(label_step["lead_time_mean_at_0.5"]),
        "val_step_catastrophic_auprc": float(metrics["label_step"]["val"]["overall"]["catastrophic_auprc"]),
        "temperature": float(metrics["training"]["temperature"]),
        "best_epoch": int(metrics["training"]["best_epoch"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_profiles = list(dict.fromkeys(str(profile) for profile in args.profiles))
    missing_required = [profile for profile in PROFILE_ORDER[1:] if profile not in requested_profiles]
    if missing_required:
        raise ValueError(f"Anti-shortcut task runs require the full profile set; missing={missing_required}")

    profile_summaries: dict[str, Any] = {}
    for profile_name in PROFILE_ORDER:
        profile_dir = output_dir / profile_name
        command = [
            sys.executable,
            "ctsc/scripts/train_precursor_model.py",
            "--dataset",
            str(Path(args.dataset)),
            "--output-dir",
            str(profile_dir),
            "--model",
            "gru",
            "--seed",
            str(int(args.split_seed)),
            "--device",
            str(args.device),
            "--feature-profile",
            profile_name,
            "--window-size",
            str(int(args.window_size)),
            "--hidden-dim",
            str(int(args.hidden_dim)),
            "--num-layers",
            str(int(args.num_layers)),
            "--dropout",
            str(float(args.dropout)),
            "--batch-size",
            str(int(args.batch_size)),
            "--epochs",
            str(int(args.epochs)),
            "--lr",
            str(float(args.lr)),
            "--weight-decay",
            str(float(args.weight_decay)),
            "--num-workers",
            str(int(args.num_workers)),
            "--step-q-low",
            str(float(args.step_q_low)),
            "--step-q-high",
            str(float(args.step_q_high)),
            "--outcome-q-low",
            str(float(args.outcome_q_low)),
            "--outcome-q-high",
            str(float(args.outcome_q_high)),
            "--critic-baseline-mode",
            "dataset",
        ]
        if args.max_train_samples is not None:
            command.extend(["--max-train-samples", str(int(args.max_train_samples))])
        if args.max_val_samples is not None:
            command.extend(["--max-val-samples", str(int(args.max_val_samples))])
        if args.max_test_samples is not None:
            command.extend(["--max-test-samples", str(int(args.max_test_samples))])
        _run(command, cwd=REPO_ROOT)
        metrics_path = profile_dir / "metrics.json"
        profile_summaries[profile_name] = {
            "metrics_path": str(metrics_path),
            "bundle_path": str(profile_dir / "precursor_model.pt"),
            **_profile_metric_summary(_load_json(metrics_path)),
        }

    payload = {
        "phase": "Unified38_current_boundary_task",
        "status": "success",
        "task_name": str(args.task_name),
        "split_seed": int(args.split_seed),
        "source_platform": _catalog_source_platform(str(args.task_name)),
        "profile_order": list(PROFILE_ORDER),
        "full_reference": dict(profile_summaries["full"]),
        "profiles": {name: dict(profile_summaries[name]) for name in PROFILE_ORDER[1:]},
        "shared_defaults": {
            "window_size": int(args.window_size),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "step_q_low": float(args.step_q_low),
            "step_q_high": float(args.step_q_high),
            "outcome_q_low": float(args.outcome_q_low),
            "outcome_q_high": float(args.outcome_q_high),
        },
    }
    payload["boundary_tag"] = _boundary_tag(payload)
    payload["interpretation_note"] = _interpretation_note(payload)
    payload["contrasts"] = {
        "full_minus_current_only_step_auprc": _relative_delta(
            float(payload["full_reference"]["step_catastrophic_auprc"]),
            float(payload["profiles"]["current_only"]["step_catastrophic_auprc"]),
        ),
        "history_without_current_cost_minus_cost_step_only_step_auprc": _relative_delta(
            float(payload["profiles"]["history_without_current_cost"]["step_catastrophic_auprc"]),
            float(payload["profiles"]["cost_step_only"]["step_catastrophic_auprc"]),
        ),
        "history_obs_action_step_only_minus_current_only_step_auprc": _relative_delta(
            float(payload["profiles"]["history_obs_action_step_only"]["step_catastrophic_auprc"]),
            float(payload["profiles"]["current_only"]["step_catastrophic_auprc"]),
        ),
    }

    summary_path = output_dir / "boundary_summary.json"
    _write_json(summary_path, payload)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "task_name": str(args.task_name),
                "split_seed": int(args.split_seed),
                "boundary_tag": payload["boundary_tag"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
