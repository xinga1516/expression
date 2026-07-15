from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.smoke


def test_main_modules_import() -> None:
    for module_name in [
        "src.dataset",
        "src.model",
        "src.utils",
        "scripts.train",
        "scripts.model_test",
        "scripts.process_data",
        "scripts.data_sanity",
        "scripts.build_sequence_assets",
        "scripts.model_compare",
    ]:
        importlib.import_module(module_name)
