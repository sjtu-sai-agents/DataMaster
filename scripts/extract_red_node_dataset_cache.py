#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_HF_HOME = "${HF_HOME}"
DEFAULT_HF_DATASETS_CACHE = f"{DEFAULT_HF_HOME}/datasets"
DEFAULT_HF_HUB_CACHE = f"{DEFAULT_HF_HOME}/hub"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def guess_repo_root(input_path: Path) -> Path:
    resolved = input_path.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for candidate in [resolved, *resolved.parents]:
        if (candidate / "runs").exists() and (candidate / "scripts").exists():
            return candidate
    return resolved


def load_config_benchmark(run_dir: Path) -> str | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception:
        return None
    inline = re.search(r"benchmark_suite:\s*\[(.*?)\]", text)
    if inline:
        first = inline.group(1).split(",")[0].strip().strip("\"'")
        return first or None
    block = re.search(r"benchmark_suite:\s*\n\s*-\s*([A-Za-z0-9_.-]+)", text)
    if block:
        return block.group(1).strip()
    return None


def guess_benchmark(run_name: str) -> str:
    name = run_name
    prefix = "math_posttrain_datatree_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    parts = name.split("_")
    stop_tokens = {
        "manual",
        "relaunch",
        "envfix",
        "noproxy",
        "toolfix",
        "toolfix2",
        "toolfix3",
        "redrestart",
        "mcpfix",
        "launch",
        "venvgpu2",
    }
    picked: list[str] = []
    for token in parts:
        if re.fullmatch(r"gpu\d+", token) or re.fullmatch(r"\d{8,}", token):
            break
        if token in stop_tokens:
            break
        picked.append(token)
    return "_".join(picked) or run_name


def find_manifest_paths(run_dir: Path) -> list[Path]:
    manifest_dir = run_dir / "workspaces" / "task_0" / "artifacts" / "manifests"
    if not manifest_dir.exists():
        return []
    return sorted(manifest_dir.glob("dataset_manifest_*.json"))


def collect_dataset_entries(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark = load_config_benchmark(run_dir) or guess_benchmark(run_dir.name)
    rows: list[dict[str, Any]] = []
    manifest_count = 0
    for manifest_path in find_manifest_paths(run_dir):
        payload = read_json(manifest_path)
        if not isinstance(payload, dict):
            continue
        manifest_count += 1
        red_node_id = str(payload.get("created_from_node") or manifest_path.stem.replace("dataset_manifest_", ""))
        for dataset in payload.get("datasets") or []:
            if not isinstance(dataset, dict):
                continue
            source_id = str(dataset.get("source_id") or "").strip()
            if not source_id:
                continue
            rows.append(
                {
                    "run_name": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "benchmark": benchmark,
                    "manifest_id": payload.get("manifest_id"),
                    "manifest_path": str(manifest_path.resolve()),
                    "red_node_id": red_node_id,
                    "search_goal": payload.get("search_goal", ""),
                    "source_id": source_id,
                    "config": str(
                        dataset.get("config")
                        or dataset.get("config_name")
                        or dataset.get("subset")
                        or dataset.get("dataset_config")
                        or ""
                    ).strip(),
                    "split": str(dataset.get("split") or "").strip(),
                    "url": dataset.get("url", ""),
                    "task_type": dataset.get("task_type", ""),
                }
            )
    summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "benchmark": benchmark,
        "manifest_count": manifest_count,
        "dataset_entry_count": len(rows),
        "unique_dataset_count": len({(row["source_id"], row["config"]) for row in rows}),
    }
    return rows, summary


def apply_hf_env(hf_home: str, hf_datasets_cache: str, hf_hub_cache: str, hf_endpoint: str) -> None:
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_hub_cache
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HF_HUB_URL"] = hf_endpoint
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    os.environ.setdefault("HF_HTTP_TIMEOUT", "300")
    for path in [hf_home, hf_datasets_cache, hf_hub_cache]:
        Path(path).mkdir(parents=True, exist_ok=True)


def prefetch_entry(
    row: dict[str, Any],
    *,
    all_configs: bool,
) -> list[dict[str, Any]]:
    from datasets import DownloadConfig, get_dataset_config_names, load_dataset, load_dataset_builder

    dataset_id = row["source_id"]
    requested_config = row["config"]
    requested_split = row["split"]
    results: list[dict[str, Any]] = []

    if requested_config:
        config_names = [requested_config]
    elif all_configs:
        try:
            config_names = [str(name) for name in (get_dataset_config_names(dataset_id) or []) if str(name).strip()]
        except Exception:
            config_names = []
        if not config_names:
            config_names = [""]
    else:
        config_names = [""]

    if not config_names:
        config_names = [""]

    download_config = DownloadConfig(resume_download=True, max_retries=4)

    for config_name in config_names:
        item = {
            "source_id": dataset_id,
            "config": config_name,
            "requested_split": requested_split,
            "status": "started",
        }
        try:
            builder = load_dataset_builder(dataset_id, name=(config_name or None))
            item["builder_name"] = getattr(builder, "builder_name", None)
            item["cache_dir"] = str(getattr(builder, "cache_dir", ""))
            item["dataset_name"] = getattr(builder, "dataset_name", None)
            builder.download_and_prepare(download_config=download_config)
            splits = list((getattr(builder.info, "splits", {}) or {}).keys())
            item["splits"] = splits
            try:
                target_split = requested_split or (splits[0] if splits else None)
                if target_split:
                    _ = load_dataset(dataset_id, name=(config_name or None), split=target_split)
                else:
                    _ = load_dataset(dataset_id, name=(config_name or None))
            except Exception as split_exc:
                item["load_warning"] = str(split_exc)
            item["status"] = "ok"
        except Exception as exc:
            item["status"] = "error"
            item["error"] = str(exc)
        results.append(item)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch datasets discovered by red nodes into the shared Hugging Face cache."
    )
    parser.add_argument("run_dirs", nargs="*", help="One or more run directories.")
    parser.add_argument("--runs-root", default=None, help="Scan this root when run_dirs is omitted.")
    parser.add_argument("--pattern", default="math_posttrain_datatree_*", help="Glob used when scanning runs-root.")
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME, help="Shared HF_HOME path.")
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE, help="Shared HF_DATASETS_CACHE path.")
    parser.add_argument("--hf-hub-cache", default=DEFAULT_HF_HUB_CACHE, help="Shared HUGGINGFACE_HUB_CACHE path.")
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT, help="HF endpoint or mirror.")
    parser.add_argument("--all-configs", action="store_true", help="Prefetch every config when manifest config is empty.")
    parser.add_argument("--limit-datasets", type=int, default=0, help="Only prefetch the first N unique dataset requests.")
    parser.add_argument("--write-summary", action="store_true", help="Also write a small summary JSON under repo cache.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_path = Path(args.run_dirs[0]).resolve() if args.run_dirs else Path.cwd()
    repo_root = guess_repo_root(seed_path)
    runs_root = Path(args.runs_root).resolve() if args.runs_root else (repo_root / "runs").resolve()

    if args.run_dirs:
        run_dirs = [Path(item).resolve() for item in args.run_dirs]
    else:
        run_dirs = sorted(path.resolve() for path in runs_root.glob(args.pattern) if path.is_dir())

    apply_hf_env(args.hf_home, args.hf_datasets_cache, args.hf_hub_cache, args.hf_endpoint)

    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows, summary = collect_dataset_entries(run_dir)
        if rows:
            all_rows.extend(rows)
            run_summaries.append(summary)

    deduped_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in all_rows:
        key = (row["source_id"], row["config"])
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)

    if args.limit_datasets and args.limit_datasets > 0:
        deduped_rows = deduped_rows[: args.limit_datasets]

    eprint("[prefetch_red_node_datasets] HF_HOME=", args.hf_home)
    eprint("[prefetch_red_node_datasets] HF_DATASETS_CACHE=", args.hf_datasets_cache)
    eprint("[prefetch_red_node_datasets] HUGGINGFACE_HUB_CACHE=", args.hf_hub_cache)
    eprint("[prefetch_red_node_datasets] HF_ENDPOINT=", args.hf_endpoint)
    eprint("[prefetch_red_node_datasets] scanned_runs=", len(run_summaries))
    eprint("[prefetch_red_node_datasets] unique_datasets=", len(deduped_rows))

    results: list[dict[str, Any]] = []
    for index, row in enumerate(deduped_rows, start=1):
        eprint(f"[{index}/{len(deduped_rows)}] prefetch {row['source_id']} config={row['config'] or '<default>'}")
        prefetched = prefetch_entry(row, all_configs=args.all_configs)
        for item in prefetched:
            item["run_name"] = row["run_name"]
            item["benchmark"] = row["benchmark"]
        results.extend(prefetched)

    ok_count = sum(1 for item in results if item["status"] == "ok")
    err_count = sum(1 for item in results if item["status"] == "error")
    summary = {
        "repo_root": str(repo_root),
        "runs_root": str(runs_root),
        "hf_home": args.hf_home,
        "hf_datasets_cache": args.hf_datasets_cache,
        "hf_hub_cache": args.hf_hub_cache,
        "hf_endpoint": args.hf_endpoint,
        "run_count": len(run_summaries),
        "unique_dataset_requests": len(deduped_rows),
        "prefetch_attempts": len(results),
        "ok_count": ok_count,
        "error_count": err_count,
        "runs": run_summaries,
        "top_datasets": Counter(item["source_id"] for item in results).most_common(50),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.write_summary:
        write_json(repo_root / "cache" / "red_node_dataset_prefetch_summary.json", {"summary": summary, "results": results})

    return 1 if err_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
