"""Matrix completion checks and compatible public command entry point."""
from __future__ import annotations
import json
from pathlib import Path
import sys
from typing import Any, Mapping
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ctsc.precursor.revision import file_sha256

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_dir(root: Path, cell: Mapping[str, Any]) -> Path:
    return root / "cell_runs" / str(cell["condition_id"]) / str(cell["task_name"]) / f"split_seed_{int(cell['split_seed'])}"


def _cell_complete(root: Path, cell: Mapping[str, Any]) -> bool:
    cell_dir = _cell_dir(root, cell)
    try:
        descriptor = _load_json(cell_dir / "cell_manifest.json")
        summary = _load_json(cell_dir / "cell_summary.json")
        if summary["status"] != "PASS" or summary["cell_sha256"] != descriptor["cell_sha256"]:
            return False
        if str(descriptor["task_name"]) != str(cell["task_name"]):
            return False
        if str(descriptor["condition_id"]) != str(cell["condition_id"]):
            return False
        if int(descriptor["split_seed"]) != int(cell["split_seed"]):
            return False
        expected_lrs = {str(key): float(value) for key, value in cell["model_learning_rates"].items()}
        if descriptor["model_learning_rates"] != expected_lrs:
            return False
        shared_records: list[dict[str, Any]] = []
        for model in cell["models"]:
            candidates = sorted((cell_dir / "models" / str(model)).glob("attempt-*/run_manifest.json"))
            valid = False
            for path in candidates:
                manifest = _load_json(path)
                protocol = manifest["protocol"]
                metrics_path = path.with_name("metrics.json")
                bundle_path = path.with_name("precursor_model.pt")
                artifacts_valid = (
                    metrics_path.is_file()
                    and bundle_path.is_file()
                    and file_sha256(metrics_path) == manifest["artifacts"]["metrics.json"]["sha256"]
                    and file_sha256(bundle_path) == manifest["artifacts"]["precursor_model.pt"]["sha256"]
                    and len(_load_json(metrics_path)["training"]["history"]) == int(manifest["training"]["epochs"])
                )
                if (
                    artifacts_valid
                    and
                    all(int(protocol[name]) == int(cell["split_seed"]) for name in ("split_seed", "sampling_seed", "train_seed", "loader_seed"))
                    and protocol["legacy_compatible"] is False
                    and manifest["training"]["loader_rng_mode"] == "explicit_loader_seed"
                    and abs(float(manifest["training"]["learning_rate"]) - expected_lrs[str(model)]) <= 1e-15
                ):
                    shared_records.append(manifest["indices"])
                    valid = True
                    break
            if not valid:
                return False
        return bool(shared_records) and all(record == shared_records[0] for record in shared_records)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


if __name__ == "__main__":
    from ctsc.scripts.run_benchmark import main
    main()
