#!/usr/bin/env python3
"""Memory Tree CLI - wraps memory_tree_interface.py

== Manifest (self) ==
    python memory_cli.py update-summary   -w /path/to/workspace -n NODE_ID --summary "..."
    python memory_cli.py append-recording -w /path/to/workspace -n NODE_ID --summary "title" --content "detail"
    python memory_cli.py modify-recording -w /path/to/workspace -n NODE_ID --rid 1 --summary "new title" --content "new detail"
    python memory_cli.py delete-recording -w /path/to/workspace -n NODE_ID --rid 1

== Read others ==
    python memory_cli.py tree             -w /path/to/workspace
    python memory_cli.py all-manifest     -w /path/to/workspace
    python memory_cli.py parent-manifest  -w /path/to/workspace -n NODE_ID
    python memory_cli.py manifest-summary -w /path/to/workspace -n NODE_ID
    python memory_cli.py manifest-all     -w /path/to/workspace -n NODE_ID
    python memory_cli.py list-children    -w /path/to/workspace -n NODE_ID

== Storage ==
    python memory_cli.py node-code        -w /path/to/workspace -n NODE_ID
    python memory_cli.py node-output      -w /path/to/workspace -n NODE_ID
    python memory_cli.py node-trajectory  -w /path/to/workspace -n NODE_ID

== Global Memory ==
    python memory_cli.py read-global      -w /path/to/workspace
    python memory_cli.py add-global       -w /path/to/workspace --summary "title" --content "detail"

== Data Link ==
    python memory_cli.py show-all-data    -w /path/to/workspace
    python memory_cli.py show-data        -w /path/to/workspace --dataset-id 1
    python memory_cli.py add-data         -w /path/to/workspace --path /abs/path --desc "description"
    python memory_cli.py add-data-record  -w /path/to/workspace --dataset-id 1 -n NODE_ID --comment "..."
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_tree_interface import (
    update_current_summary,
    append_current_recordings,
    modify_current_recordings,
    delete_current_recordings,
    get_current_tree,
    get_all_manifest,
    get_parent_manifest,
    get_manifest_summary,
    get_manifest_all,
    list_children,
    get_node_code,
    get_node_output,
    get_node_trajectory,
    read_global_memory,
    add_global_memory,
    show_all_data,
    show_detailed_data,
    add_new_data,
    add_data_record,
)


def main():
    parser = argparse.ArgumentParser(description="Memory Tree CLI")
    parser.add_argument("-w", "--workspace", required=True, help="Workspace directory path")
    sub = parser.add_subparsers(dest="command")

    # === Manifest (self) ===

    p = sub.add_parser("update-summary", help="Update current node TL;DR")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")
    p.add_argument("--summary", required=True, help="New TL;DR summary")

    p = sub.add_parser("append-recording", help="Append a new recording")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")
    p.add_argument("--summary", required=True, help="Recording title")
    p.add_argument("--content", required=True, help="Recording detail")

    p = sub.add_parser("modify-recording", help="Modify an existing recording")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")
    p.add_argument("--rid", type=int, required=True, help="Recording ID")
    p.add_argument("--summary", default=None, help="New summary (None=keep)")
    p.add_argument("--content", default=None, help="New content (None=keep)")

    p = sub.add_parser("delete-recording", help="Delete a recording")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")
    p.add_argument("--rid", type=int, required=True, help="Recording ID")

    # === Read others ===

    sub.add_parser("tree", help="Show full memory tree structure")

    sub.add_parser("all-manifest", help="Get TL;DR of all nodes")

    p = sub.add_parser("parent-manifest", help="Get parent node manifest")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    p = sub.add_parser("manifest-summary", help="Get node TL;DR + recording titles")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    p = sub.add_parser("manifest-all", help="Get full manifest content")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    p = sub.add_parser("list-children", help="List child nodes")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    # === Storage ===

    p = sub.add_parser("node-code", help="Get node code.py")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    p = sub.add_parser("node-output", help="Get node stdout.txt")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    p = sub.add_parser("node-trajectory", help="Get node trajectory.json")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")

    # === Global Memory ===

    sub.add_parser("read-global", help="Read global memory")

    p = sub.add_parser("add-global", help="Add global memory entry")
    p.add_argument("--summary", required=True, help="Memory title")
    p.add_argument("--content", required=True, help="Memory content")

    # === Data Link ===

    sub.add_parser("show-all-data", help="Show all dataset summaries")

    p = sub.add_parser("show-data", help="Show detailed dataset info")
    p.add_argument("--dataset-id", type=int, required=True, help="Dataset ID")

    p = sub.add_parser("add-data", help="Add new dataset record")
    p.add_argument("--path", required=True, help="Dataset absolute path")
    p.add_argument("--desc", required=True, help="Initial description")

    p = sub.add_parser("add-data-record", help="Add comment to dataset")
    p.add_argument("--dataset-id", type=int, required=True, help="Dataset ID")
    p.add_argument("-n", "--node-id", required=True, help="Node ID")
    p.add_argument("--comment", required=True, help="Comment text")

    args = parser.parse_args()
    w = args.workspace

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Manifest (self)
    if args.command == "update-summary":
        print(update_current_summary(workspace=w, node_id=args.node_id, summary=args.summary))
    elif args.command == "append-recording":
        print(append_current_recordings(workspace=w, node_id=args.node_id,
              recording_summary=args.summary, recording_content=args.content))
    elif args.command == "modify-recording":
        print(modify_current_recordings(workspace=w, node_id=args.node_id,
              recording_id=args.rid, recording_summary=args.summary, recording_content=args.content))
    elif args.command == "delete-recording":
        print(delete_current_recordings(workspace=w, node_id=args.node_id, recording_id=args.rid))

    # Read others
    elif args.command == "tree":
        print(get_current_tree(workspace=w))
    elif args.command == "all-manifest":
        print(get_all_manifest(workspace=w))
    elif args.command == "parent-manifest":
        print(get_parent_manifest(workspace=w, node_id=args.node_id))
    elif args.command == "manifest-summary":
        print(get_manifest_summary(workspace=w, node_id=args.node_id))
    elif args.command == "manifest-all":
        print(get_manifest_all(workspace=w, node_id=args.node_id))
    elif args.command == "list-children":
        print(list_children(workspace=w, node_id=args.node_id))

    # Storage
    elif args.command == "node-code":
        print(get_node_code(workspace=w, node_id=args.node_id))
    elif args.command == "node-output":
        print(get_node_output(workspace=w, node_id=args.node_id))
    elif args.command == "node-trajectory":
        print(get_node_trajectory(workspace=w, node_id=args.node_id))

    # Global Memory
    elif args.command == "read-global":
        print(read_global_memory(workspace=w))
    elif args.command == "add-global":
        print(add_global_memory(workspace=w, memory_summary=args.summary, memory_content=args.content))

    # Data Link
    elif args.command == "show-all-data":
        print(show_all_data(workspace=w))
    elif args.command == "show-data":
        print(show_detailed_data(workspace=w, dataset_id=args.dataset_id))
    elif args.command == "add-data":
        print(add_new_data(workspace=w, dataset_path=args.path, init_descriptions=args.desc))
    elif args.command == "add-data-record":
        print(add_data_record(workspace=w, dataset_id=args.dataset_id,
              node_id=args.node_id, comment=args.comment))


if __name__ == "__main__":
    main()
