#!/usr/bin/env python3
"""Operate Submission CLI - wraps for_datanode.py

== Code Management ==
    python operate_cli.py read-code      -w /path/to/workspace -n NODE_ID
    python operate_cli.py write-code     -w /path/to/workspace -n NODE_ID --code "..."
    python operate_cli.py write-code     -w /path/to/workspace -n NODE_ID --code-file /path/to/file.py
    python operate_cli.py fix-code       -w /path/to/workspace -n NODE_ID --old "..." --new "..."

== Execution ==
    python operate_cli.py run-code       -w /path/to/workspace -n NODE_ID [--timeout 3600]

== Validation & Grading ==
    python operate_cli.py validate       -w /path/to/workspace -n NODE_ID
    python operate_cli.py grade          -w /path/to/workspace -n NODE_ID [--timeout 300]
"""
import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from for_datanode import (
    read_code,
    write_code,
    fix_code,
    run_code,
    validate_submission,
    grade_code,
)


def main():
    parser = argparse.ArgumentParser(description="Operate Submission CLI")
    parser.add_argument("-w", "--workspace", required=True, help="Workspace directory path")
    parser.add_argument("-n", "--node-id", required=True, help="Node ID")
    sub = parser.add_subparsers(dest="command")

    # === Code Management ===

    sub.add_parser("read-code", help="Read all code components (base + dataloader + template)")

    p = sub.add_parser("write-code", help="Write DataLoader code")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--code", help="Code string to write")
    g.add_argument("--code-file", help="Path to code file to read and write")
    p.add_argument("--override", action="store_true", help="Override existing code")

    p = sub.add_parser("fix-code", help="Fix DataLoader code via string replacement")
    p.add_argument("--old", required=True, help="Old string to replace")
    p.add_argument("--new", required=True, help="New string")
    p.add_argument("--replace-all", action="store_true", help="Replace all occurrences")

    # === Execution ===

    p = sub.add_parser("run-code", help="Execute code (auto-assembles all components)")
    p.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")

    # === Validation & Grading ===

    sub.add_parser("validate", help="Validate submission file")

    p = sub.add_parser("grade", help="Grade submission file")
    p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")

    args = parser.parse_args()
    w = args.workspace
    n = args.node_id

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Code Management
    if args.command == "read-code":
        print(read_code(node_id=n, workspace=w))
    elif args.command == "write-code":
        code = args.code
        if args.code_file:
            with open(args.code_file, "r", encoding="utf-8") as f:
                code = f.read()
        print(write_code(code=code, node_id=n, workspace=w, override=args.override))
    elif args.command == "fix-code":
        print(fix_code(old_string=args.old, new_string=args.new,
              node_id=n, workspace=w, replace_all=args.replace_all))

    # Execution
    elif args.command == "run-code":
        print(asyncio.run(run_code(node_id=n, workspace=w, timeout=args.timeout)))

    # Validation & Grading
    elif args.command == "validate":
        print(asyncio.run(validate_submission(node_id=n, workspace=w)))
    elif args.command == "grade":
        print(asyncio.run(grade_code(node_id=n, workspace=w, timeout=args.timeout)))


if __name__ == "__main__":
    main()
