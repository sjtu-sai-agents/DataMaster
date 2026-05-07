#!/usr/bin/env python3
from pathlib import Path
import argparse
import re


def remove_existing_ablation_block(text: str) -> str:
    """
    Remove an existing top-level ablation block without deleting later top-level YAML keys.
    Assumes standard YAML where top-level keys start at column 0.
    """
    lines = text.splitlines()
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if re.match(r"^ablation:\s*$", line):
            i += 1

            while i < len(lines):
                next_line = lines[i]

                # Keep skipping blank lines inside/after the block.
                if next_line.strip() == "":
                    i += 1
                    continue

                # Stop when another top-level key begins.
                if re.match(r"^[A-Za-z0-9_\"'-]+:\s*", next_line):
                    break

                # Otherwise this line belongs to the ablation block.
                i += 1

            continue

        out.append(line)
        i += 1

    return "\n".join(out).rstrip() + "\n"


def update_ablation_block(
    text: str,
    use_red_node: bool,
    use_memory: bool,
    black_mode: str = "agent",
) -> str:
    text = remove_existing_ablation_block(text)

    text += f"""

# ============================================
# Component Ablation 配置
# ============================================
ablation:
  use_red_node: {str(use_red_node).lower()}
  use_memory: {str(use_memory).lower()}
  black_mode: {black_mode}
"""
    return text


def generate_for_competition(comp: str, config_root: Path) -> None:
    root = config_root / comp
    base = root / f"config_{comp}.yaml"

    if not base.exists():
        raise FileNotFoundError(f"Base config not found: {base}")

    text = base.read_text(encoding="utf-8")

    no_red = root / f"config_{comp}_no_red.yaml"
    no_red.write_text(
        update_ablation_block(
            text,
            use_red_node=False,
            use_memory=True,
            black_mode="agent",
        ),
        encoding="utf-8",
    )
    print(f"created: {no_red}")

    no_memory = root / f"config_{comp}_no_memory.yaml"
    no_memory.write_text(
        update_ablation_block(
            text,
            use_red_node=True,
            use_memory=False,
            black_mode="agent",
        ),
        encoding="utf-8",
    )
    print(f"created: {no_memory}")

    rule_black = root / f"config_{comp}_rule_black.yaml"
    rule_black.write_text(
        update_ablation_block(
            text,
            use_red_node=True,
            use_memory=True,
            black_mode="rule_based",
        ),
        encoding="utf-8",
    )
    print(f"created: {rule_black}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default=None)
    parser.add_argument(
        "--config-root",
        default="configs/ml_master_datatree/yaml_configs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate ablation configs for all competition folders.",
    )
    args = parser.parse_args()

    config_root = Path(args.config_root)

    if args.all:
        for root in sorted(config_root.iterdir()):
            if not root.is_dir():
                continue

            comp = root.name
            base = root / f"config_{comp}.yaml"
            if not base.exists():
                print(f"skip: {comp}, missing {base}")
                continue

            generate_for_competition(comp, config_root)
        return

    if not args.competition:
        raise ValueError("Please provide --competition COMP or use --all")

    generate_for_competition(args.competition, config_root)


if __name__ == "__main__":
    main()