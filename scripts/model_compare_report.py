"""Manifest-driven collection layer for the consolidated model-comparison report."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_EXCLUDE = {"seed", "exp_name", "resume", "git_hash"}
METRIC_SPECS: dict[str, tuple[str, str]] = {
    "val_loss_ema": ("val_loss_ema", "min"),
    "val_loss": ("val_loss", "min"),
    "val_rmse": ("val_rmse", "min"),
    "val_pearson": ("val_pearson_all", "max"),
    "val_spearman": ("val_spearman_all", "max"),
}


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as string-keyed rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: object) -> float | None:
    """Return a finite float or None."""
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested dictionaries for a report-wide raw run table."""
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(flatten(name, child))
    return flattened


def stable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove run-specific fields before grouping otherwise identical configurations."""
    return {key: config[key] for key in sorted(config) if key not in CONFIG_EXCLUDE}


def config_group(config: dict[str, Any]) -> str:
    """Create a stable short identifier for a configuration excluding seed."""
    encoded = json.dumps(stable_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def best_log_row(rows: Iterable[dict[str, str]], column: str, direction: str) -> tuple[int | None, float | None]:
    """Return epoch/value for the best finite row under a monitor direction."""
    best_epoch: int | None = None
    best_value: float | None = None
    for row in rows:
        value = number(row.get(column))
        epoch_value = number(row.get("epoch"))
        if value is None or epoch_value is None:
            continue
        if best_value is None or (direction == "min" and value < best_value) or (direction == "max" and value > best_value):
            best_epoch, best_value = int(epoch_value), value
    return best_epoch, best_value


def checkpoint_audit(run_dir: Path, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract checkpoint selection and test provenance from one completed run."""
    train_log_path = run_dir / "log" / "train_log.csv"
    train_log = read_csv(train_log_path) if train_log_path.exists() else []
    monitor, direction = METRIC_SPECS.get(str(config.get("checkpoint_metric")), ("val_loss_ema", "min"))
    best_epoch, best_value = best_log_row(train_log, monitor, direction)
    audit: dict[str, Any] = {
        "validation_checkpoint_metric": config.get("checkpoint_metric", "val_loss_ema"),
        "validation_monitor_column": monitor,
        "best_model_epoch": best_epoch,
        "best_model_validation_value": best_value,
        "test_checkpoint": Path(str(metrics.get("checkpoint", "best_model.safetensors"))).name,
        "test_metrics": "mse, rmse, pearson_r, spearman_r",
    }
    for label, column, metric_direction in (
        ("best_val_rmse", "val_rmse", "min"),
        ("best_val_pearson", "val_pearson_all", "max"),
        ("best_val_spearman", "val_spearman_all", "max"),
    ):
        epoch, value = best_log_row(train_log, column, metric_direction)
        audit[f"{label}_epoch"] = epoch
        audit[label] = value
    return audit


def resolve_path(project_root: Path, value: str) -> Path:
    """Resolve a manifest path relative to the project root."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root / candidate


def required_columns(entry: dict[str, Any]) -> set[str]:
    """Return required source columns for one manifest artifact."""
    return {str(column) for column in entry.get("required_columns", [])}


def validate_artifact(project_root: Path, entry: dict[str, Any]) -> tuple[Path, list[dict[str, str]]]:
    """Validate one registered CSV artifact and return parsed rows."""
    path = resolve_path(project_root, str(entry["path"]))
    if not path.exists():
        raise FileNotFoundError(f"Required model-comparison artifact is missing: {path}")
    rows = read_csv(path)
    headers = set(rows[0]) if rows else set()
    missing = required_columns(entry) - headers
    if missing:
        raise ValueError(f"Artifact {path} is missing required columns: {sorted(missing)}")
    return path, rows


def collect_stage_runs(project_root: Path, stage: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect run-level raw rows and checkpoint audit values for one stage."""
    stage_name = str(stage["id"])
    root = resolve_path(project_root, str(stage["run_root"]))
    if not root.exists():
        raise FileNotFoundError(f"Stage run root is missing: {root}")
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "test" / "test_metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = read_json(config_path)
        metrics = read_json(metrics_path)
        audit = checkpoint_audit(run_dir, config, metrics)
        row: dict[str, Any] = {
            "stage": stage_name,
            "run": run_dir.name,
            "config_group": config_group(config),
            "seed": config.get("seed"),
        }
        row.update(flatten("config", config))
        row.update(flatten("test", metrics))
        row.update(flatten("checkpoint", audit))
        rows.append(row)
    return rows


def merge_stage_summary_rows(raw_rows: list[dict[str, Any]], artifact_id: str, rows: list[dict[str, str]]) -> None:
    """Attach stage-summary fields to matching raw run rows when a run key exists."""
    by_key = {str(row.get("run")): row for row in raw_rows}
    for summary_row in rows:
        run_key = summary_row.get("run") or summary_row.get("exp_name") or summary_row.get("run_name")
        if run_key not in by_key:
            continue
        target = by_key[run_key]
        for key, value in summary_row.items():
            target[f"summary.{artifact_id}.{key}"] = value


def union_columns(rows: list[dict[str, Any]], leading: tuple[str, ...] = ()) -> list[str]:
    """Produce stable union columns while preserving useful leading identifiers."""
    remaining = sorted({key for row in rows for key in row} - set(leading))
    return [key for key in leading if any(key in row for row in rows)] + remaining


def seed_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize test metrics for groups identical except for training seed."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["stage"]), str(row["config_group"]))].append(row)
    summaries: list[dict[str, Any]] = []
    metrics = ("test.mse", "test.rmse", "test.pearson_r", "test.spearman_r")
    for (stage, group_id), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "stage": stage,
            "config_group": group_id,
            "n_runs": len(rows),
            "seeds": ", ".join(str(row.get("seed")) for row in sorted(rows, key=lambda item: str(item.get("seed")))),
            "model": rows[0].get("config.model"),
            "loss": rows[0].get("config.loss_type", rows[0].get("config.loss")),
            "sequence_column": rows[0].get("config.sequence_column"),
            "contrastive_weight": rows[0].get("config.contrastive_weight", 0),
        }
        for metric in metrics:
            values = [number(row.get(metric)) for row in rows]
            valid = [value for value in values if value is not None]
            label = metric.removeprefix("test.")
            summary[f"{label}_mean"] = mean(valid) if valid else None
            summary[f"{label}_std"] = stdev(valid) if len(valid) > 1 else None
        summaries.append(summary)
    return summaries


def collect_report(project_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect and validate all report inputs into a renderer-neutral payload."""
    inventory: list[dict[str, Any]] = []
    stages_payload: list[dict[str, Any]] = []
    all_raw_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for stage in manifest["stages"]:
        raw_rows = collect_stage_runs(project_root, stage)
        stage_tables: list[dict[str, Any]] = []
        for entry in stage.get("summary_tables", []):
            path, rows = validate_artifact(project_root, entry)
            artifact_id = str(entry["id"])
            inventory.append({"stage": stage["id"], "kind": "summary_table", "id": artifact_id, "path": str(path), "status": "validated"})
            merge_stage_summary_rows(raw_rows, artifact_id, rows)
            stage_tables.append({"id": artifact_id, "title": entry.get("title", artifact_id), "rows": rows})
        for entry in stage.get("paired_statistics", []):
            path, rows = validate_artifact(project_root, entry)
            inventory.append({"stage": stage["id"], "kind": "paired_statistics", "id": entry["id"], "path": str(path), "status": "validated"})
            for row in rows:
                paired_rows.append({"stage": stage["id"], "artifact": entry["id"], **row})
        figures: list[dict[str, str]] = []
        for entry in stage.get("figures", []):
            path = resolve_path(project_root, str(entry["path"]))
            if not path.exists():
                raise FileNotFoundError(f"Required stage figure is missing: {path}")
            inventory.append({"stage": stage["id"], "kind": "figure", "id": entry["id"], "path": str(path), "status": "validated"})
            figures.append({"id": str(entry["id"]), "title": str(entry.get("title", entry["id"])), "path": str(path)})
        stages_payload.append({
            "id": stage["id"],
            "title": stage.get("title", stage["id"]),
            "raw_rows": raw_rows,
            "raw_columns": union_columns(raw_rows, ("stage", "run", "seed", "config_group")),
            "tables": stage_tables,
            "figures": figures,
        })
        all_raw_rows.extend(raw_rows)
    return {
        "manifest_version": manifest.get("version", "unknown"),
        "inventory": inventory,
        "run_audit_rows": all_raw_rows,
        "run_audit_columns": union_columns(all_raw_rows, ("stage", "run", "seed", "config_group", "checkpoint.validation_checkpoint_metric", "checkpoint.best_model_epoch", "checkpoint.test_checkpoint", "test.mse", "test.rmse", "test.pearson_r", "test.spearman_r")),
        "seed_rows": seed_summary(all_raw_rows),
        "paired_rows": paired_rows,
        "stages": stages_payload,
    }


def refresh_report(project_root: Path, manifest: dict[str, Any]) -> None:
    """Run only the explicitly registered comparison workflows before reporting."""
    for entry in manifest.get("refresh", []):
        workflow = str(entry["workflow"])
        arguments = [str(value) for value in entry.get("args", [])]
        subprocess.run(
            [sys.executable, str(project_root / "scripts" / "model_compare.py"), workflow, *arguments],
            cwd=project_root,
            check=True,
        )
