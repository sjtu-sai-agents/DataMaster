#!/usr/bin/env python
"""
.venv/bin/python scripts/prefetch_models.py \
  --manifest configs/data_master/prefetch_models.json \
  --env-from-mcp configs/data_master/mcp_config4data_aptos2019.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _eprint(*args) -> None:
    print(*args, file=sys.stderr)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_env_from_mcp_config(path: Path) -> None:
    cfg = _read_json(path)
    env = (
        cfg.get("mcpServers", {})
        .get("operate_submission", {})
        .get("env", {})
    )
    if not isinstance(env, dict):
        return
    for k, v in env.items():
        if v is None:
            continue
        os.environ[str(k)] = str(v)

    # Normalize common HuggingFace cache envs so different libs agree.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_cache = os.path.join(hf_home, "hub")
        os.environ.setdefault("HF_HUB_CACHE", hub_cache)
        # Transformers will honor TRANSFORMERS_CACHE; pointing it at HF_HOME keeps everything together.
        os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
        # Some tooling reads HF_ASSETS_CACHE as well.
        os.environ.setdefault("HF_ASSETS_CACHE", os.path.join(hf_home, "assets"))

    # If user set datasets cache, keep it explicit.
    # (datasets uses HF_DATASETS_CACHE; it is independent from model hub cache.)
    if os.environ.get("HF_DATASETS_CACHE"):
        os.environ.setdefault("HF_DATASETS_CACHE", os.environ["HF_DATASETS_CACHE"])


def _ensure_dirs() -> None:
    # Respect caller-provided env. If not set, don't guess.
    for k in ["HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"]:
        v = os.environ.get(k)
        if v:
            Path(v).mkdir(parents=True, exist_ok=True)


def _iter_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def prefetch_timm(models: list[str]) -> int:
    import timm  # noqa: WPS433

    failed = 0
    for name in models:
        try:
            _eprint(f"[timm] prefetch {name}")
            m = timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
            # free memory quickly
            del m
        except Exception as exc:
            failed += 1
            _eprint(f"[timm] FAILED {name}: {exc}")
    return failed


def prefetch_torchvision(models: list[str]) -> int:
    import torch  # noqa: WPS433
    import torchvision  # noqa: WPS433

    failed = 0

    # Classification backbones (handle both old/new torchvision APIs)
    for name in models:
        if name == "fasterrcnn_resnet50_fpn":
            continue
        try:
            _eprint(f"[torchvision] prefetch {name}")
            fn = getattr(torchvision.models, name)
            try:
                # New API: weights=...
                weights_enum = getattr(torchvision.models, f"{name.upper()}_Weights", None)
                if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
                    m = fn(weights=weights_enum.DEFAULT)
                else:
                    m = fn(pretrained=True)
            except TypeError:
                m = fn(pretrained=True)
            m.eval()
            with torch.no_grad():
                _ = m(torch.zeros(1, 3, 224, 224))
            del m
        except Exception as exc:
            failed += 1
            _eprint(f"[torchvision] FAILED {name}: {exc}")

    # Detection model
    if "fasterrcnn_resnet50_fpn" in models:
        try:
            _eprint("[torchvision] prefetch fasterrcnn_resnet50_fpn")
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn  # noqa: WPS433
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights  # noqa: WPS433

                m = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            except Exception:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn  # noqa: WPS433

                m = fasterrcnn_resnet50_fpn(pretrained=True)
            m.eval()
            del m
        except Exception as exc:
            failed += 1
            _eprint(f"[torchvision] FAILED fasterrcnn_resnet50_fpn: {exc}")

    return failed


def prefetch_transformers(models: list[str]) -> int:
    from transformers import (  # noqa: WPS433
        AutoConfig,
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        T5ForConditionalGeneration,
        T5Tokenizer,
    )

    failed = 0
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None

    for name in models:
        try:
            _eprint(f"[transformers] prefetch {name}")
            # Config + tokenizer always
            _ = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
            try:
                _ = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, use_fast=True)
            except TypeError:
                _ = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

            # Model type depends on task; try a few common heads then fallback to AutoModel
            model = None
            for ctor in (AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModel):
                try:
                    model = ctor.from_pretrained(name, cache_dir=cache_dir)
                    break
                except Exception:
                    continue
            if model is None:
                raise RuntimeError("could not load model with common AutoModel* classes")
            del model
        except Exception as exc:
            failed += 1
            _eprint(f"[transformers] FAILED {name}: {exc}")

    # Special-case t5-small (some initial_code imports direct T5 classes)
    if "t5-small" in models:
        try:
            _eprint("[transformers] prefetch t5-small (direct classes)")
            _ = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir)
            _ = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir=cache_dir)
        except Exception as exc:
            failed += 1
            _eprint(f"[transformers] FAILED t5-small direct: {exc}")

    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=str,
        default="configs/data_master/prefetch_models.json",
        help="Path to prefetch_models.json",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "timm", "torchvision", "transformers"],
        help="Prefetch only one source",
    )
    ap.add_argument(
        "--env-from-mcp",
        type=str,
        default="",
        help="Optional. Path to mcp_config4data_*.json; loads operate_submission.env into current process.",
    )
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        _eprint(f"manifest not found: {manifest_path}")
        return 2

    if args.env_from_mcp:
        mcp_path = Path(args.env_from_mcp)
        if not mcp_path.exists():
            _eprint(f"mcp config not found: {mcp_path}")
            return 2
        _apply_env_from_mcp_config(mcp_path)

    _ensure_dirs()
    manifest = _read_json(manifest_path)

    timm_models = _iter_unique(manifest.get("timm_models", []))
    tv_models = _iter_unique(manifest.get("torchvision_models", []))
    tf_models = _iter_unique(manifest.get("transformers_models", []))

    _eprint("=== Prefetch settings ===")
    _eprint("HF_ENDPOINT=", os.environ.get("HF_ENDPOINT"))
    _eprint("HF_HOME=", os.environ.get("HF_HOME"))
    _eprint("HF_HUB_CACHE=", os.environ.get("HF_HUB_CACHE"))
    _eprint("HF_DATASETS_CACHE=", os.environ.get("HF_DATASETS_CACHE"))
    _eprint("TRANSFORMERS_CACHE=", os.environ.get("TRANSFORMERS_CACHE"))
    _eprint("TORCH_HOME=", os.environ.get("TORCH_HOME"))
    _eprint("=========================")

    total_failed = 0
    if args.only in ("all", "timm"):
        total_failed += prefetch_timm(timm_models)
    if args.only in ("all", "torchvision"):
        total_failed += prefetch_torchvision(tv_models)
    if args.only in ("all", "transformers"):
        total_failed += prefetch_transformers(tf_models)

    if total_failed:
        _eprint(f"Done with failures: {total_failed}")
        return 1
    _eprint("Done. All prefetches succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

