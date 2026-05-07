#!/usr/bin/env python3
"""
Generate PDF plots from all gallery examples.

This script runs all Python scripts in gallery_python_new and saves
the output as PDF files in the images directory.
"""

import os
import sys
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add the gallery directories to path
base_dir = Path(__file__).parent
gallery_dir = base_dir / "gallery_python_new"
output_dir = base_dir / "images"
output_dir.mkdir(exist_ok=True)

# Load custom style
style_file = base_dir / "mplstyle" / "times_new_roman.mplstyle"
plt.style.use(str(style_file))

# Custom color palette
CUSTOM_COLORS = [
    '#05c6b4',  # cyan
    '#00b4cd',  # light blue
    '#009edd',  # blue
    '#0082db',  # dark blue
    '#6f60c0',  # purple
    '#99358e',  # magenta
]

# Set the color cycle for matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CUSTOM_COLORS)
plt.rcParams['lines.color'] = CUSTOM_COLORS[0]
plt.rcParams['patch.facecolor'] = CUSTOM_COLORS[0]


def modify_script_content(content: str, output_file: Path) -> str:
    """
    Modify script content to replace plt.show() with plt.savefig().

    Args:
        content: Original script content
        output_file: Path to output PDF file

    Returns:
        Modified script content
    """
    lines = content.split('\n')
    modified_lines = []

    # Track if we've already replaced a show() call
    show_replaced = False

    for line in lines:
        # Skip lines that already have savefig (to avoid duplicates)
        if 'fig.savefig(' in line or 'plt.savefig(' in line:
            # Replace existing savefig with our output path
            if 'fig.savefig(' in line:
                modified_lines.append(
                    f"fig.savefig('{output_file}', bbox_inches='tight', dpi=150)"
                )
            elif 'plt.savefig(' in line:
                modified_lines.append(
                    f"plt.savefig('{output_file}', bbox_inches='tight', dpi=150)"
                )
            continue

        # Replace plt.show() with savefig
        if 'plt.show()' in line and not show_replaced:
            modified_lines.append(
                f"plt.savefig('{output_file}', bbox_inches='tight', dpi=150)"
            )
            show_replaced = True
        else:
            modified_lines.append(line)

    # If no plt.show() was found, add savefig at the end before the last line
    if not show_replaced:
        # Find the last non-empty line
        for i in range(len(modified_lines) - 1, -1, -1):
            if modified_lines[i].strip() and not modified_lines[i].strip().startswith('#'):
                modified_lines.insert(
                    i + 1,
                    f"plt.savefig('{output_file}', bbox_inches='tight', dpi=150)"
                )
                break

    return '\n'.join(modified_lines)


def run_single_script(script_path: Path) -> bool:
    """
    Run a single script and capture its plt.show() to save as PDF.

    Args:
        script_path: Path to the Python script

    Returns:
        True if successful, False otherwise
    """
    script_name = script_path.stem
    output_file = output_dir / f"{script_name}.pdf"

    print(f"Processing: {script_path.relative_to(base_dir)}")

    try:
        # Read and modify the script
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create a modified version that saves instead of shows
        modified_content = modify_script_content(content, output_file)

        # Write to a temp file and execute
        temp_file = script_path.parent / f"temp_{script_name}.py"

        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        # Execute in a fresh namespace
        namespace = {'__name__': '__main__', '__file__': str(temp_file)}

        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                code = compile(f.read(), str(temp_file), 'exec')
            exec(code, namespace)

            # Close all figures to free memory
            plt.close('all')

            # Check if file was created
            if output_file.exists():
                print(f"  ✓ Saved: {output_file.name}")
                return True
            else:
                print(f"  ✗ File not created")
                return False

        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        # Close all figures to free memory
        try:
            plt.close('all')
        except:
            pass
        return False


def main():
    """Generate PDF plots for all examples."""
    print("="*60)
    print("Matplotlib Gallery - PDF Generator")
    print("="*60)
    print(f"\nSource: {gallery_dir}")
    print(f"Output: {output_dir}")
    print(f"Style: Times New Roman")
    print(f"Colors: Custom palette")
    print()

    # Get all Python files
    py_files = sorted(list(gallery_dir.rglob("*.py")))

    print(f"Found {len(py_files)} Python files\n")

    success_count = 0
    fail_count = 0

    for py_file in py_files:
        if run_single_script(py_file):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed:  {fail_count}")
    print(f"  Total:     {len(py_files)}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
