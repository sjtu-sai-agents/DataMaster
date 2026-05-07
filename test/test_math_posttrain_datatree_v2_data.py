from __future__ import annotations

import json
import os
import sys
import types
import importlib.util
from pathlib import Path


def _load_data_utils():
    root = Path(__file__).resolve().parents[1]
    package_name = "_math_posttrain_datatree_v2_utils_for_test"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / "playground/math_posttrain_datatree_v2/core/utils")]
        sys.modules[package_name] = package

    module_name = f"{package_name}.data"
    module_path = root / "playground/math_posttrain_datatree_v2/core/utils/data.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


data_utils = _load_data_utils()


def test_materialize_dataset_entry_uses_configured_hf_cache(monkeypatch, tmp_path: Path) -> None:
    hf_cache = tmp_path / "hf_cache"
    (hf_cache / "unit___cached_math").mkdir(parents=True)
    calls: list[tuple[str, str]] = []

    def fake_get_dataset_config_names(dataset_id: str) -> list[str]:
        assert dataset_id == "unit/cached_math"
        return []

    def fake_get_dataset_split_names(**kwargs) -> list[str]:
        assert kwargs == {"path": "unit/cached_math"}
        return ["train"]

    def fake_load_dataset(dataset_id: str, *, split: str):
        assert dataset_id == "unit/cached_math"
        assert os.environ["HF_DATASETS_CACHE"] == str(hf_cache)
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        calls.append((dataset_id, split))
        return [
            {
                "problem": "What is 2+3?",
                "solution": "2+3=5",
                "final_answer": "5",
            }
        ]

    monkeypatch.setattr(data_utils, "_HF_SANDBOX_URL", "")
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("HF_DATASETS_OFFLINE", raising=False)
    monkeypatch.delenv("FORCE_ONLINE", raising=False)
    monkeypatch.setattr("datasets.get_dataset_config_names", fake_get_dataset_config_names)
    monkeypatch.setattr("datasets.get_dataset_split_names", fake_get_dataset_split_names)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    local_path = data_utils.materialize_dataset_entry(
        {
            "source_id": "unit/cached_math",
            "url": "https://huggingface.co/datasets/unit/cached_math",
        },
        tmp_path / "materialized",
        max_rows=8,
        data_access_config={
            "hf_cache": str(hf_cache),
            "datasets_server": {"enabled": False},
        },
    )

    rows = [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert calls == [("unit/cached_math", "train[:8]")]
    assert rows[0]["problem"] == "What is 2+3?"
