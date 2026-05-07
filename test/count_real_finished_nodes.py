#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


NODE_ID_RE = re.compile(r"[a-f0-9]{32}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config_exp_id(run_dir: Path) -> str:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return "unknown"

    text = config_path.read_text(encoding="utf-8")
    m = re.search(r'^\s*exp_id:\s*["\']?([^"\']+)["\']?\s*$', text, re.MULTILINE)
    if m:
        return m.group(1).strip()

    return "unknown"


def find_trajectory_file(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "trajectories").glob("task_*/trajectory.json"))
    if not candidates:
        raise FileNotFoundError(f"No trajectory.json found under {run_dir / 'trajectories'}")

    if len(candidates) > 1:
        print("[WARN] Multiple trajectory files found. Using the first one:")
        for p in candidates:
            print(f"  - {p}")

    return candidates[0]


def load_entries(path: Path) -> List[Dict[str, Any]]:
    obj = read_json(path)

    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        # fallback for possible alternative formats
        if isinstance(obj.get("trajectories"), list):
            return [x for x in obj["trajectories"] if isinstance(x, dict)]
        return [obj]

    return []


def get_text_fields(entry: Dict[str, Any]) -> List[str]:
    fields = []

    for key in ["agent_id", "task_id", "agent_name", "exp_name", "status"]:
        val = entry.get(key)
        if isinstance(val, str):
            fields.append(val)

    return fields


def is_metric_entry(entry: Dict[str, Any]) -> bool:
    text = " ".join(get_text_fields(entry)).lower()
    return "metric" in text


def infer_stage(entry: Dict[str, Any]) -> str:
    text = " ".join(get_text_fields(entry)).lower()

    if "metric" in text:
        return "metric"
    if "initial" in text:
        return "initial"
    if "black" in text:
        return "black"
    if "red" in text:
        return "red"

    return "unknown"


def infer_node_id(entry: Dict[str, Any]) -> Optional[str]:
    """
    Prefer task_id / agent_id because they usually contain the true 32-char node id.
    Example:
      2d6177a85b8a4181aff99d2f2238951a_initial
      2d6177a85b8a4181aff99d2f2238951a_initial_initial_worker_0
    """
    for key in ["task_id", "agent_id", "exp_name", "node_id", "id"]:
        val = entry.get(key)
        if not isinstance(val, str):
            continue

        m = NODE_ID_RE.search(val)
        if m:
            return m.group(0)

    return None


def analyze_trajectory(trajectory_path: Path, run_dir: Optional[Path]) -> Dict[str, Any]:
    entries = load_entries(trajectory_path)

    node_stage: Dict[str, str] = {}
    node_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    ignored_metric_entries = 0
    ignored_no_node_id = 0

    raw_stage_counter = Counter()
    counted_stage_counter = Counter()

    for entry in entries:
        stage = infer_stage(entry)
        raw_stage_counter[stage] += 1

        if is_metric_entry(entry):
            ignored_metric_entries += 1
            continue

        node_id = infer_node_id(entry)
        if node_id is None:
            ignored_no_node_id += 1
            continue

        node_stage[node_id] = stage
        node_entries[node_id].append(entry)
        counted_stage_counter[stage] += 1

    nodes_by_stage: Dict[str, List[str]] = defaultdict(list)
    for node_id, stage in node_stage.items():
        nodes_by_stage[stage].append(node_id)

    for stage in nodes_by_stage:
        nodes_by_stage[stage] = sorted(set(nodes_by_stage[stage]))

    competition = "unknown"
    if run_dir is not None:
        competition = read_config_exp_id(run_dir)

    return {
        "competition": competition,
        "trajectory_path": str(trajectory_path),
        "run_dir": str(run_dir) if run_dir else None,
        "total_trajectory_entries": len(entries),
        "ignored_metric_entries": ignored_metric_entries,
        "ignored_no_node_id_entries": ignored_no_node_id,
        "counted_non_metric_entries": len(entries) - ignored_metric_entries - ignored_no_node_id,
        "unique_real_nodes": len(node_stage),
        "raw_entries_by_stage": dict(raw_stage_counter),
        "counted_entries_by_stage": dict(counted_stage_counter),
        "unique_nodes_by_stage": {
            stage: len(nodes)
            for stage, nodes in sorted(nodes_by_stage.items())
        },
        "node_ids_by_stage": dict(sorted(nodes_by_stage.items())),
    }


def print_report(result: Dict[str, Any], show_ids: bool = False) -> None:
    line = "=" * 90
    print("\n" + line)
    print(f"competition:                {result['competition']}")
    print(f"run_dir:                    {result['run_dir']}")
    print(f"trajectory:                 {result['trajectory_path']}")
    print(line)

    print(f"total trajectory entries:    {result['total_trajectory_entries']}")
    print(f"ignored metric entries:      {result['ignored_metric_entries']}")
    print(f"ignored no-node-id entries:  {result['ignored_no_node_id_entries']}")
    print(f"counted non-metric entries:  {result['counted_non_metric_entries']}")
    print()
    print(f"real total nodes:            {result['unique_real_nodes']}")

    print("\nRaw entries by stage:")
    for stage, count in sorted(result["raw_entries_by_stage"].items()):
        print(f"  {stage:<10}: {count}")

    print("\nCounted entries by stage:")
    for stage, count in sorted(result["counted_entries_by_stage"].items()):
        print(f"  {stage:<10}: {count}")

    print("\nUnique real nodes by stage:")
    for stage, count in sorted(result["unique_nodes_by_stage"].items()):
        print(f"  {stage:<10}: {count}")

    if show_ids:
        print("\nNode ids by stage:")
        for stage, node_ids in result["node_ids_by_stage"].items():
            print(f"\n[{stage}]")
            for node_id in node_ids:
                print(f"  {node_id}")

    print(line)


def save_json(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count real generated nodes directly from trajectory.json."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=str, help="Run directory containing trajectories/task_0/trajectory.json")
    group.add_argument("--trajectory", type=str, help="Direct path to trajectory.json")

    parser.add_argument("--show-ids", action="store_true", help="Print all node ids by stage.")
    parser.add_argument("--save-json", type=str, default=None, help="Optional JSON output path.")

    args = parser.parse_args()

    run_dir: Optional[Path] = None

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists():
            raise SystemExit(f"run_dir not found: {run_dir}")
        trajectory_path = find_trajectory_file(run_dir)
    else:
        trajectory_path = Path(args.trajectory).expanduser().resolve()
        if not trajectory_path.exists():
            raise SystemExit(f"trajectory file not found: {trajectory_path}")

        # Infer run_dir from .../runs/<run_name>/trajectories/task_0/trajectory.json
        try:
            run_dir = trajectory_path.parents[2]
            if run_dir.name == "trajectories":
                run_dir = None
        except Exception:
            run_dir = None

    result = analyze_trajectory(trajectory_path, run_dir)
    print_report(result, show_ids=args.show_ids)

    if args.save_json:
        out = Path(args.save_json).expanduser().resolve()
        save_json(result, out)
        print(f"\n[INFO] saved json report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())