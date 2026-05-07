#!/usr/bin/env python3
"""
Convert trajectory from new format (_append_trajectory_entry_old) to old format (_append_trajectory_entry)

Source format: Each step is a separate entry
Target format: Each agent has one entry with accumulated messages
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

SOURCE_FILE = "${PROJECT_ROOT}/runs/ml_master_datatree_20260326_165235/trajectories/task_0/trajectory.json"
OUTPUT_FILE = "${PROJECT_ROOT}/runs/ml_master_datatree_20260326_165235/trajectories/task_0/trajectory_converted.json"


def convert_trajectory(source_entries: List[Dict]) -> List[Dict]:
    """Convert from new format to old format

    Source: Multiple entries per agent (one per step)
    Target: Single entry per agent with accumulated messages
    """
    # Group by agent_id
    agent_entries: Dict[str, Dict] = {}

    for entry in source_entries:
        traj = entry.get("trajectory", {})
        task_id = traj.get("task_id", "unknown")
        agent_name = traj.get("agent_name", "unknown")
        agent_id = f"{task_id}_{agent_name}"

        if agent_id not in agent_entries:
            # Initialize with first entry
            agent_entries[agent_id] = {
                "agent_id": agent_id,
                "exp_name": entry.get("exp_name", ""),
                "exp_index": entry.get("exp_index", 0),
                "status": entry.get("status", "running"),
                "agent_name": agent_name,
                "task_id": task_id,
                "steps": entry.get("steps", 0),
                "trajectory": {
                    "messages": [],
                    "meta": {
                        "agent_version": "1.0.0",
                        "agent_name": agent_name,
                        "step": entry.get("steps", 0),
                        "start_time": traj.get("start_time"),
                        "end_time": traj.get("end_time"),
                        "status": entry.get("status", "running"),
                    }
                }
            }

        # Merge messages from dialogs and steps
        dialogs = traj.get("dialogs", [])
        steps = traj.get("steps", [])

        # Add dialog messages (system/user prompts)
        for dialog in dialogs:
            messages = dialog.get("messages", [])
            for msg in messages:
                # Check if message already exists (avoid duplicates)
                if msg not in agent_entries[agent_id]["trajectory"]["messages"]:
                    agent_entries[agent_id]["trajectory"]["messages"].append(msg)

        # Add assistant message from steps
        for step in steps:
            assistant_msg = step.get("assistant_message", {})
            if assistant_msg and assistant_msg not in agent_entries[agent_id]["trajectory"]["messages"]:
                agent_entries[agent_id]["trajectory"]["messages"].append(assistant_msg)

            # Add tool responses
            for tool_resp in step.get("tool_responses", []):
                if tool_resp not in agent_entries[agent_id]["trajectory"]["messages"]:
                    agent_entries[agent_id]["trajectory"]["messages"].append(tool_resp)

        # Update step count and status with latest
        agent_entries[agent_id]["steps"] = entry.get("steps", 0)
        agent_entries[agent_id]["status"] = entry.get("status", "running")
        agent_entries[agent_id]["trajectory"]["meta"]["step"] = entry.get("steps", 0)
        agent_entries[agent_id]["trajectory"]["meta"]["status"] = entry.get("status", "running")

    return list(agent_entries.values())


def main():
    print(f"Reading source file: {SOURCE_FILE}")

    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        source_entries = json.load(f)

    print(f"Found {len(source_entries)} entries in source format")

    # Convert
    converted = convert_trajectory(source_entries)

    print(f"Converted to {len(converted)} entries in target format")

    # Save
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, default=str, ensure_ascii=False)

    print(f"Saved to: {OUTPUT_FILE}")

    # Print sample
    print("\nSample converted entry:")
    print(json.dumps(converted[0], indent=2, default=str, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
