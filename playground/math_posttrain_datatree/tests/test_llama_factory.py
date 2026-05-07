from __future__ import annotations

import json

from playground.math_posttrain_datatree.core.utils.llama_factory import render_recipe


def test_render_recipe_accepts_common_override_keys_and_batch_size_alias(tmp_path):
    dataset_path = tmp_path / "alpaca_train.jsonl"
    dataset_path.write_text(
        '{"instruction":"i","input":"","output":"o"}\n',
        encoding="utf-8",
    )
    recipe_path = tmp_path / "recipe.json"
    output_dir = tmp_path / "out"

    render_recipe(
        recipe_path,
        base_model="/data/public_model/Qwen3-1.7B",
        dataset_path=dataset_path,
        output_dir=output_dir,
        overrides={
            "batch_size": 2,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "learning_rate": 5e-5,
        },
    )

    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert payload["per_device_train_batch_size"] == 2
    assert "batch_size" not in payload
    assert payload["warmup_ratio"] == 0.1
    assert payload["weight_decay"] == 0.01
    assert payload["learning_rate"] == 5e-5


def test_render_recipe_persists_cutoff_len(tmp_path):
    dataset_path = tmp_path / "alpaca_train.jsonl"
    dataset_path.write_text(
        '{"instruction":"i","input":"","output":"o"}\n',
        encoding="utf-8",
    )
    recipe_path = tmp_path / "recipe.json"
    output_dir = tmp_path / "out"

    render_recipe(
        recipe_path,
        base_model="/data/public_model/Qwen3-1.7B",
        dataset_path=dataset_path,
        output_dir=output_dir,
        overrides={"cutoff_len": 4096},
    )

    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert payload["cutoff_len"] == 4096
