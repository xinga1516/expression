"""Unified public CLI for promoter and 3'UTR sequence asset construction.

Use one of the workflows below instead of invoking the implementation scripts
directly. Existing implementation filenames remain available for compatibility
with historical commands and remote job logs.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORKFLOWS = {
    "full": "scripts.build_promoter_stage1_assets",
    "reuse": "scripts.build_reused_split_sequence_assets",
    "utr": "scripts.build_utr_stage_assets",
}


def parse_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Build sequence assets with a single public entry point."
    )
    parser.add_argument("workflow", choices=sorted(WORKFLOWS))
    parsed, forwarded = parser.parse_known_args(argv)
    return str(parsed.workflow), list(forwarded)


def dispatch(workflow: str, forwarded_args: Sequence[str]) -> None:
    """Delegate a workflow to its compatible implementation module."""
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown sequence-asset workflow: {workflow}")
    module = importlib.import_module(WORKFLOWS[workflow])
    previous_argv = sys.argv
    try:
        sys.argv = [f"build_sequence_assets {workflow}", *forwarded_args]
        if hasattr(module, "main"):
            module.main()
        else:
            module.build_assets(module.parse_args())
    finally:
        sys.argv = previous_argv


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] in WORKFLOWS:
        dispatch(tokens[0], tokens[1:])
        return
    workflow, forwarded = parse_args(tokens)
    dispatch(workflow, forwarded)


if __name__ == "__main__":
    main()
