from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts import build_sequence_assets, model_compare


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("module", "workflow", "expected_backend"),
    [
        (build_sequence_assets, "full", "scripts.build_promoter_stage1_assets"),
        (build_sequence_assets, "reuse", "scripts.build_reused_split_sequence_assets"),
        (model_compare, "stage1-ablation", "scripts.stage1_training_ablation"),
        (model_compare, "stage1-bootstrap", "scripts.summarize_stage1_bootstrap"),
        (model_compare, "stage2-ablation", "scripts.stage2_contrastive_ablation"),
    ],
)
def test_public_entrypoint_dispatches_and_restores_argv(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    workflow: str,
    expected_backend: str,
) -> None:
    called: dict[str, object] = {}

    def fake_main() -> None:
        called["argv"] = list(sys.argv)

    def fake_import(name: str) -> SimpleNamespace:
        called["backend"] = name
        return SimpleNamespace(main=fake_main)

    original_argv = list(sys.argv)
    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    module.dispatch(workflow, ["--example", "value"])

    assert called["backend"] == expected_backend
    assert called["argv"] == [f"{module.__name__.split('.')[-1]} {workflow}", "--example", "value"]
    assert sys.argv == original_argv


def test_asset_entrypoint_supports_build_assets_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def fake_parse_args() -> str:
        called["argv"] = list(sys.argv)
        return "parsed"

    def fake_build_assets(args: str) -> None:
        called["args"] = args

    monkeypatch.setattr(
        build_sequence_assets.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            parse_args=fake_parse_args,
            build_assets=fake_build_assets,
        ),
    )
    build_sequence_assets.dispatch("full", ["--example", "value"])

    assert called["argv"] == ["build_sequence_assets full", "--example", "value"]
    assert called["args"] == "parsed"


def test_model_compare_report_dispatches_to_report_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, object] = {}

    def fake_report(arguments: list[str]) -> None:
        called["arguments"] = arguments

    monkeypatch.setattr(model_compare, "dispatch_report", fake_report)
    model_compare.dispatch("report", ["--output", "outputs/comparison.xlsx"])

    assert called["arguments"] == ["--output", "outputs/comparison.xlsx"]
