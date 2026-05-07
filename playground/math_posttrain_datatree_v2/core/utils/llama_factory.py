from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io import write_json
from .types import TrainResult

logger = logging.getLogger(__name__)

ALLOWED_HPARAM_OVERRIDES = {
    "num_train_epochs",
    "learning_rate",
    "cutoff_len",
    "batch_size",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "weight_decay",
    "max_samples",
}


def finetuning_type_from_training_mode(training_mode: str | None) -> str:
    mode = str(training_mode or "lora_sft").strip().lower()
    if mode in {"full", "full_sft", "sft_full"}:
        return "full"
    if mode in {"lora", "lora_sft", "sft_lora"}:
        return "lora"
    raise ValueError(f"unsupported training_mode: {training_mode}")


def _absolute_path_preserve_symlink(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _resolve_env_bin_dir(
    *,
    python_bin: str | Path | None = None,
    env_dir: str | Path | None = None,
) -> Path | None:
    if python_bin:
        return _absolute_path_preserve_symlink(python_bin).parent
    if env_dir:
        candidate = _absolute_path_preserve_symlink(env_dir) / "bin"
        if candidate.exists():
            return candidate
    return None


def resolve_llamafactory_command(
    *,
    python_bin: str | Path | None = None,
    env_dir: str | Path | None = None,
) -> list[str] | None:
    explicit_env_requested = python_bin is not None or env_dir is not None
    bin_dir = _resolve_env_bin_dir(python_bin=python_bin, env_dir=env_dir)
    if bin_dir is not None:
        cli_path = bin_dir / "llamafactory-cli"
        if cli_path.exists():
            return [str(cli_path), "train"]
        python_path = _absolute_path_preserve_symlink(python_bin) if python_bin else bin_dir / "python"
        if python_path.exists():
            probe = subprocess.run(
                [str(python_path), "-c", "import importlib.util; print(importlib.util.find_spec('llamafactory') is not None)"],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip() == "True":
                return [str(python_path), "-m", "llamafactory.cli", "train"]
        if explicit_env_requested:
            return None
    elif explicit_env_requested:
        return None

    cli = shutil.which("llamafactory-cli")
    if cli:
        return [cli, "train"]

    module_spec = importlib.util.find_spec("llamafactory")
    if module_spec is not None:
        return [sys.executable, "-m", "llamafactory.cli", "train"]

    return None


def ensure_llamafactory_available(
    *,
    python_bin: str | Path | None = None,
    env_dir: str | Path | None = None,
) -> tuple[bool, str]:
    command = resolve_llamafactory_command(python_bin=python_bin, env_dir=env_dir)
    if command is None:
        return (
            False,
            "LLaMA Factory is not available. Install `llamafactory`, expose `llamafactory-cli`, or point to a dedicated env.",
        )
    return True, " ".join(command)


def validate_alpaca_dataset(dataset_path: str | Path) -> tuple[bool, str]:
    path = Path(dataset_path)
    if not path.exists():
        return False, f"dataset file not found: {path}"
    required = {"instruction", "input", "output"}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        import json

        payload = json.loads(line)
        missing = required - payload.keys()
        if missing:
            return False, f"line {lineno} missing fields: {sorted(missing)}"
    return True, "ok"


def infer_template_from_model(base_model: str) -> str:
    model_name = base_model.lower()
    if "qwen3" in model_name:
        return "qwen3"
    if "qwen" in model_name:
        return "qwen"
    if "llama-3" in model_name or "llama3" in model_name:
        return "llama3"
    if "llama" in model_name:
        return "llama"
    if "mistral" in model_name:
        return "mistral"
    return "default"


def create_dataset_registry(
    dataset_path: str | Path,
    *,
    dataset_name: str = "smoke_alpaca",
) -> tuple[Path, str]:
    dataset_path = Path(dataset_path).resolve()
    dataset_dir = dataset_path.parent
    registry_path = dataset_dir / "dataset_info.json"
    payload = {
        dataset_name: {
            "file_name": dataset_path.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    write_json(registry_path, payload)
    return registry_path, dataset_name


def normalize_hparam_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(overrides or {})
    if "batch_size" in normalized and "per_device_train_batch_size" not in normalized:
        normalized["per_device_train_batch_size"] = normalized.pop("batch_size")
    else:
        normalized.pop("batch_size", None)
    return normalized


def render_recipe(
    recipe_path: str | Path,
    *,
    base_model: str,
    dataset_path: str | Path,
    output_dir: str | Path,
    overrides: dict[str, Any] | None = None,
    template_override: str | None = None,
    training_mode: str | None = None,
) -> Path:
    recipe_path = Path(recipe_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    output_dir = Path(output_dir).resolve()
    _, dataset_name = create_dataset_registry(dataset_path)
    overrides = normalize_hparam_overrides(overrides)
    unexpected = sorted(set(overrides) - ALLOWED_HPARAM_OVERRIDES)
    if unexpected:
        raise ValueError(f"unsupported override keys: {unexpected}")
    template = template_override or infer_template_from_model(base_model)
    finetuning_type = finetuning_type_from_training_mode(training_mode)
    payload = {
        "stage": "sft",
        "model_name_or_path": base_model,
        "do_train": True,
        "dataset": dataset_name,
        "dataset_dir": str(dataset_path.parent),
        "finetuning_type": finetuning_type,
        "template": template,
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "logging_steps": 1,
        "save_steps": 1000,
        "plot_loss": False,
        **overrides,
    }
    return write_json(recipe_path, payload)


def _resolve_python_executable(
    *,
    python_bin: str | Path | None = None,
    env_dir: str | Path | None = None,
) -> Path | None:
    if python_bin is not None:
        return _absolute_path_preserve_symlink(python_bin)
    bin_dir = _resolve_env_bin_dir(env_dir=env_dir)
    if bin_dir is not None:
        candidate = bin_dir / "python"
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def merge_lora_adapter_for_eval(
    *,
    adapter_dir: str | Path,
    merged_dir: str | Path,
    base_model: str,
    log_path: str | Path,
    python_bin: str | Path | None = None,
    env_dir: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[bool, str]:
    adapter_dir = Path(adapter_dir).resolve()
    merged_dir = Path(merged_dir).resolve()
    log_path = Path(log_path)
    python_path = _resolve_python_executable(python_bin=python_bin, env_dir=env_dir)
    if python_path is None or not python_path.exists():
        message = f"merge python executable not found for adapter export: {python_bin or env_dir or sys.executable}"
        log_path.write_text(message + "\n", encoding="utf-8")
        return False, message

    merge_script = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "import torch",
            "from peft import AutoPeftModelForCausalLM",
            "from transformers import AutoTokenizer",
            "adapter_dir = Path(sys.argv[1]).resolve()",
            "merged_dir = Path(sys.argv[2]).resolve()",
            "base_model = sys.argv[3]",
            "device_map = 'auto' if torch.cuda.is_available() else 'cpu'",
            "model = AutoPeftModelForCausalLM.from_pretrained(",
            "    str(adapter_dir),",
            "    torch_dtype='auto',",
            "    low_cpu_mem_usage=True,",
            "    device_map=device_map,",
            ")",
            "model = model.merge_and_unload()",
            "merged_dir.mkdir(parents=True, exist_ok=True)",
            "model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size='2GB')",
            "tokenizer_source = str(adapter_dir) if (adapter_dir / 'tokenizer_config.json').exists() else base_model",
            "tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)",
            "tokenizer.save_pretrained(str(merged_dir))",
            "print(str(merged_dir))",
        ]
    )

    env = os.environ.copy()
    preferred_bin = _resolve_env_bin_dir(python_bin=python_bin, env_dir=env_dir)
    active_bin = preferred_bin or python_path.parent
    env["PATH"] = str(active_bin) + os.pathsep + env.get("PATH", "")
    if env_overrides:
        env.update(env_overrides)

    proc = subprocess.run(
        [str(python_path), "-c", merge_script, str(adapter_dir), str(merged_dir), base_model],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return False, f"adapter merge export failed with code {proc.returncode}"
    if not (merged_dir / "config.json").exists():
        return False, f"merged model directory missing config.json: {merged_dir}"
    return True, str(merged_dir)


def run_llama_factory_sft(
    *,
    dataset_path: str | Path,
    recipe_path: str | Path,
    output_dir: str | Path,
    base_model: str,
    overrides: dict[str, Any] | None = None,
    template_override: str | None = None,
    dry_run: bool = True,
    env_overrides: dict[str, str] | None = None,
    python_bin: str | Path | None = None,
    env_dir: str | Path | None = None,
    merge_for_evaluation: bool = False,
    training_mode: str | None = None,
) -> TrainResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = render_recipe(
        recipe_path,
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        overrides=overrides,
        template_override=template_override,
        training_mode=training_mode,
    )
    log_path = output_dir / "train.log"
    checkpoint_path = output_dir
    command_parts = resolve_llamafactory_command(python_bin=python_bin, env_dir=env_dir)
    explicit_env_requested = python_bin is not None or env_dir is not None
    command = (
        " ".join(command_parts + [str(recipe_path)])
        if command_parts is not None
        else f"llamafactory-cli train {recipe_path}"
    )

    if dry_run:
        log_path.write_text(
            "dry_run=true\n"
            f"base_model={base_model}\n"
            f"dataset={dataset_path}\n"
            f"recipe={recipe_path}\n",
            encoding="utf-8",
        )
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        result = TrainResult(
            status="dry_run",
            checkpoint_path=str(checkpoint_path),
            recipe_path=str(recipe_path),
            train_log_path=str(log_path),
            command=command,
            dry_run=True,
            metrics={"epochs": overrides.get("num_train_epochs", 1) if overrides else 1},
        )
        write_json(output_dir / "train_result.json", result.to_dict())
        return result

    if command_parts is None:
        reasons: list[str] = []
        if python_bin is not None:
            python_path = _absolute_path_preserve_symlink(python_bin)
            if not python_path.exists():
                reasons.append(f"configured python_bin not found: {python_path}")
        if env_dir is not None:
            env_path = _absolute_path_preserve_symlink(env_dir)
            if not env_path.exists():
                reasons.append(f"configured env_dir not found: {env_path}")
        if not reasons and explicit_env_requested:
            reasons.append("llamafactory is not installed in the configured environment")
        if not reasons:
            reasons.append("llamafactory-cli is not available in the active environment")
        failure_message = "\n".join(reasons) + "\n"
        log_path.write_text(failure_message, encoding="utf-8")
        result = TrainResult(
            status="failed",
            checkpoint_path=str(checkpoint_path),
            recipe_path=str(recipe_path),
            train_log_path=str(log_path),
            command=command,
            dry_run=False,
            metrics={"returncode": None, "reason": "; ".join(reasons)},
        )
        write_json(output_dir / "train_result.json", result.to_dict())
        return result

    env = os.environ.copy()
    preferred_bin = _resolve_env_bin_dir(python_bin=python_bin, env_dir=env_dir)
    active_bin = preferred_bin or Path(sys.executable).parent
    env["PATH"] = str(active_bin) + os.pathsep + env.get("PATH", "")
    if env_overrides:
        env.update(env_overrides)

    logger.info("=" * 80)
    logger.info("🚀 TRAINING STARTED")
    logger.info("=" * 80)
    logger.info("Command: %s", " ".join(command_parts + [str(Path(recipe_path).resolve())]))
    logger.info("Model: %s", base_model)
    logger.info("Dataset: %s", dataset_path)
    logger.info("Output: %s", output_dir)
    logger.info("Recipe: %s", recipe_path)
    logger.info("Log file: %s", log_path)
    logger.info("=" * 80)
    logger.info("Training is running in background. Monitor progress with:")
    logger.info("  tail -f %s", output_dir / "trainer_log.jsonl")
    logger.info("  watch -n 1 nvidia-smi")
    logger.info("=" * 80)

    proc = subprocess.run(
        command_parts + [str(Path(recipe_path).resolve())],
        cwd=output_dir,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETED" if proc.returncode == 0 else "❌ TRAINING FAILED")
    logger.info("=" * 80)
    logger.info("Return code: %s", proc.returncode)
    logger.info("Log written to: %s", log_path)
    logger.info("=" * 80)

    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    status = "completed" if proc.returncode == 0 else "failed"
    metrics: dict[str, Any] = {"returncode": proc.returncode}
    should_merge_lora = (
        proc.returncode == 0
        and merge_for_evaluation
        and finetuning_type_from_training_mode(training_mode) == "lora"
        and (output_dir / "adapter_config.json").exists()
    )
    if should_merge_lora:
        merged_dir = output_dir / "merged_model"
        merge_log_path = output_dir / "merge.log"
        merged_ok, merged_value = merge_lora_adapter_for_eval(
            adapter_dir=output_dir,
            merged_dir=merged_dir,
            base_model=base_model,
            log_path=merge_log_path,
            python_bin=python_bin,
            env_dir=env_dir,
            env_overrides=env_overrides,
        )
        if merged_ok:
            checkpoint_path = Path(merged_value)
            metrics["merged_model_path"] = merged_value
            metrics["merge_log_path"] = str(merge_log_path)
        else:
            status = "failed"
            metrics["merge_error"] = merged_value
            metrics["merge_log_path"] = str(merge_log_path)
    result = TrainResult(
        status=status,
        checkpoint_path=str(checkpoint_path),
        recipe_path=str(recipe_path),
        train_log_path=str(log_path),
        command=command,
        dry_run=False,
        metrics=metrics,
    )
    write_json(output_dir / "train_result.json", result.to_dict())
    return result
