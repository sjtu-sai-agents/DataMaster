"""Data preview generation utilities for ML-Master"""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate(input_dir: str | Path, max_files: int = 10) -> str:
    """Generate a data preview for the agent.

    Args:
        input_dir: Path to the input directory containing data files
        max_files: Maximum number of files to preview

    Returns:
        A formatted string preview of the data
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        return f"Input directory not found: {input_dir}"

    preview_parts = []
    preview_parts.append(f"# Data Preview\n")
    preview_parts.append(f"Input directory: {input_path}\n")

    # List all files
    all_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(input_path)
            all_files.append((file_path, rel_path))

    preview_parts.append(f"Total files found: {len(all_files)}\n")

    # Preview each file
    for i, (file_path, rel_path) in enumerate(all_files[:max_files]):
        preview_parts.append(f"\n## File {i+1}: {rel_path}")
        preview_parts.append(f"Full path: {file_path}")
        preview_parts.append(f"Size: {file_path.stat().st_size} bytes")

        # Try to preview file contents
        try:
            file_preview = _preview_file(file_path)
            if file_preview:
                preview_parts.append(f"\n{file_preview}")
        except Exception as e:
            preview_parts.append(f"\nError previewing file: {e}")

    if len(all_files) > max_files:
        preview_parts.append(f"\n... and {len(all_files) - max_files} more files")

    return "\n".join(preview_parts)


def _preview_file(file_path: Path, max_lines: int = 20) -> str | None:
    """Preview a single file.

    Args:
        file_path: Path to the file
        max_lines: Maximum lines to show

    Returns:
        A string preview of the file
    """
    suffix = file_path.suffix.lower()

    # CSV files
    if suffix in ['.csv']:
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)
            return f"```\n{df.head()}\n```\n\nColumns: {list(df.columns)}\nShape: {df.shape}"
        except Exception as e:
            return f"Error reading CSV: {e}"

    # JSON files
    elif suffix in ['.json', '.jsonl']:
        try:
            with open(file_path, 'r') as f:
                if suffix == '.jsonl':
                    lines = [f.readline() for _ in range(3)]
                    data = [json.loads(line) for line in lines if line.strip()]
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        data = data[:3]

            preview = json.dumps(data, indent=2)[:2000]
            return f"```json\n{preview}\n```"
        except Exception as e:
            return f"Error reading JSON: {e}"

    # Text files
    elif suffix in ['.txt', '.md', '.py']:
        try:
            with open(file_path, 'r') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
            content = '\n'.join(lines)
            if len(lines) >= max_lines:
                content += "\n... (truncated)"
            return f"```\n{content}\n```"
        except Exception as e:
            return f"Error reading text: {e}"

    # Other files
    else:
        return f"Binary or unsupported file type: {suffix}"


def generate_for_task(workspace_path: str | Path) -> str:
    """Generate a data preview specifically for a Kaggle-like task.

    Args:
        workspace_path: Path to the workspace

    Returns:
        A formatted data preview string
    """
    workspace = Path(workspace_path)
    input_dir = workspace / "input"

    if not input_dir.exists():
        return f"No input directory found at: {input_dir}"

    return generate(input_dir)
