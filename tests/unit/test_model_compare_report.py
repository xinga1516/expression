from __future__ import annotations

import json

import pytest

from scripts import model_compare_report
from scripts.model_compare_report import collect_report, read_json


pytestmark = pytest.mark.unit


def _write_run(root, name: str, seed: int, mse: float) -> None:
    run_dir = root / name
    (run_dir / "test").mkdir(parents=True)
    (run_dir / "log").mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "exp_name": name,
                "seed": seed,
                "model": "CNNFlattenPromoterModel",
                "loss_type": "mse",
                "sequence_column": "sequence",
                "checkpoint_metric": "val_rmse",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "test" / "test_metrics.json").write_text(
        json.dumps(
            {
                "checkpoint": str(run_dir / "checkpoints" / "best_model.safetensors"),
                "mse": mse,
                "rmse": mse**0.5,
                "pearson_r": 0.2 + seed / 1000,
                "spearman_r": 0.1 + seed / 1000,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "log" / "train_log.csv").write_text(
        "epoch,val_rmse,val_pearson_all,val_spearman_all\n0,2.0,0.1,0.1\n3,1.0,0.2,0.2\n",
        encoding="utf-8",
    )


def _manifest() -> dict[str, object]:
    return {
        "version": "test",
        "stages": [
            {
                "id": "stage1",
                "title": "Stage 1",
                "run_root": "runs",
                "summary_tables": [
                    {
                        "id": "runs",
                        "path": "summary/runs.csv",
                        "required_columns": ["run", "mse"],
                    }
                ],
                "paired_statistics": [
                    {
                        "id": "paired",
                        "path": "summary/paired.csv",
                        "required_columns": ["comparison", "level", "n_pairs", "mean_delta"],
                    }
                ],
                "figures": [{"id": "figure", "path": "summary/figure.png"}],
            }
        ],
        "refresh": [],
    }


def test_collect_report_builds_raw_seed_and_paired_summaries(tmp_path) -> None:
    _write_run(tmp_path / "runs", "run_seed1", 1, 4.0)
    _write_run(tmp_path / "runs", "run_seed7", 7, 9.0)
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "runs.csv").write_text("run,mse\nrun_seed1,4.0\nrun_seed7,9.0\n", encoding="utf-8")
    (summary_dir / "paired.csv").write_text(
        "comparison,level,n_pairs,mean_delta\nmodel_vs_control,per_gene,2,0.1\n",
        encoding="utf-8",
    )
    (summary_dir / "figure.png").write_bytes(b"png")

    report = collect_report(tmp_path, _manifest())

    assert len(report["run_audit_rows"]) == 2
    assert len(report["seed_rows"]) == 1
    assert report["seed_rows"][0]["seeds"] == "1, 7"
    assert report["paired_rows"] == [
        {
            "stage": "stage1",
            "artifact": "paired",
            "comparison": "model_vs_control",
            "level": "per_gene",
            "n_pairs": "2",
            "mean_delta": "0.1",
        }
    ]
    raw_row = report["stages"][0]["raw_rows"][0]
    assert raw_row["checkpoint.best_model_epoch"] == 3
    assert raw_row["summary.runs.mse"] == "4.0"


def test_collect_report_rejects_missing_required_columns(tmp_path) -> None:
    (tmp_path / "runs").mkdir()
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir()
    (summary_dir / "runs.csv").write_text("run\nmissing\n", encoding="utf-8")
    (summary_dir / "paired.csv").write_text("comparison,level,n_pairs,mean_delta\nx,per_gene,2,0.1\n", encoding="utf-8")
    (summary_dir / "figure.png").write_bytes(b"png")

    with pytest.raises(ValueError, match="missing required columns"):
        collect_report(tmp_path, _manifest())


def test_read_json_requires_object(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        read_json(path)


def test_refresh_report_runs_only_manifest_registered_workflows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], cwd: object, check: bool) -> None:
        commands.append(command)
        assert cwd == tmp_path
        assert check is True

    monkeypatch.setattr(model_compare_report.subprocess, "run", fake_run)
    model_compare_report.refresh_report(
        tmp_path,
        {"refresh": [{"workflow": "stage1-bootstrap", "args": ["--stage1-dir", "outputs/stage1"]}]},
    )

    assert commands == [
        [
            model_compare_report.sys.executable,
            str(tmp_path / "scripts" / "model_compare.py"),
            "stage1-bootstrap",
            "--stage1-dir",
            "outputs/stage1",
        ]
    ]
