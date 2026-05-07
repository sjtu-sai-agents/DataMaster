from __future__ import annotations

from playground.math_posttrain_datatree.core.exp.black_exp import (
    DEFAULT_NPROC_PER_NODE,
    resolve_llama_factory_runtime,
)


def test_resolve_llama_factory_runtime_defaults_to_single_process():
    python_bin, env_dir, env_overrides = resolve_llama_factory_runtime(None)

    assert python_bin is None
    assert env_dir is None
    assert env_overrides["NPROC_PER_NODE"] == str(DEFAULT_NPROC_PER_NODE)
    assert "CUDA_VISIBLE_DEVICES" not in env_overrides


def test_resolve_llama_factory_runtime_accepts_explicit_gpu_settings():
    python_bin, env_dir, env_overrides = resolve_llama_factory_runtime(
        {
            "python_bin": "./.venv_llamafactory/bin/python",
            "env_dir": "./.venv_llamafactory",
            "nproc_per_node": 1,
            "cuda_visible_devices": "0",
        }
    )

    assert python_bin == "./.venv_llamafactory/bin/python"
    assert env_dir == "./.venv_llamafactory"
    assert env_overrides["NPROC_PER_NODE"] == "1"
    assert env_overrides["CUDA_VISIBLE_DEVICES"] == "0"
