"""Unified public CLI for model-comparison and model-ablation workflows."""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORKFLOWS = {
    "stage1-ablation": "scripts.stage1_training_ablation",
    "stage1-bootstrap": "scripts.summarize_stage1_bootstrap",
    "stage1-sequence-interaction": "scripts.summary_stage1",
    "stage2-summary": "scripts.summarize_stage2",
    "stage2-gene-balanced-motifs": "scripts.summarize_gene_balanced_motifs",
    "stage2-ablation": "scripts.stage2_contrastive_ablation",
    "report": None,
}
REPORT_SCRIPT = PROJECT_ROOT / "scripts" / "model_compare_workbook.mjs"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "model_compare_report.json"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "outputs" / "model_compare" / "model_compare_summary.xlsx"


def parse_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run model-comparison, motif-summary, or ablation workflows."
    )
    parser.add_argument("workflow", choices=sorted(WORKFLOWS))
    parsed, forwarded = parser.parse_known_args(argv)
    return str(parsed.workflow), list(forwarded)


def dispatch(workflow: str, forwarded_args: Sequence[str]) -> None:
    """Delegate a comparison workflow to its stable implementation module."""
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown model-comparison workflow: {workflow}")
    if workflow == "report":
        dispatch_report(forwarded_args)
        return
    module = importlib.import_module(WORKFLOWS[workflow])
    previous_argv = sys.argv
    try:
        sys.argv = [f"model_compare {workflow}", *forwarded_args]
        module.main()
    finally:
        sys.argv = previous_argv


def parse_report_args(forwarded_args: Sequence[str]) -> argparse.Namespace:
    """Parse arguments for the manifest-driven report workflow."""
    parser = argparse.ArgumentParser(
        description="Validate completed comparison artifacts and export one consolidated xlsx report."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Refresh only manifest-declared comparison workflows first.")
    return parser.parse_args(list(forwarded_args))


def dispatch_report(forwarded_args: Sequence[str]) -> None:
    """Collect manifest-declared artifacts and render one xlsx report through Node."""
    from scripts.model_compare_report import collect_report, read_json, refresh_report

    args = parse_report_args(forwarded_args)
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest = read_json(manifest_path)
    if args.refresh:
        refresh_report(PROJECT_ROOT, manifest)
    payload = collect_report(PROJECT_ROOT, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_binary = os.environ.get("MODEL_COMPARE_NODE") or shutil.which("node")
    if node_binary is None:
        raise RuntimeError(
            "The report renderer requires Node.js. Set MODEL_COMPARE_NODE to the "
            "Node executable that has @oai/artifact-tool available."
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json_path = Path(handle.name)
        import json

        json.dump(payload, handle, ensure_ascii=False)
    try:
        subprocess.run(
            [node_binary, str(REPORT_SCRIPT), "--payload", str(json_path), "--output", str(output_path)],
            check=True,
        )
    finally:
        json_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in WORKFLOWS:
        dispatch(tokens[0], tokens[1:])
        return
    workflow, forwarded = parse_args(tokens)
    dispatch(workflow, forwarded)


if __name__ == "__main__":
    main()
