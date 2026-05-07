"""
Documentation Generator Module

This module is responsible for reading code example information from GALLERY_DOCS_RAW.md,
reading corresponding Python source code files, and generating complete documentation to README4AGENTS.md
"""

import os
import re
from typing import List, Dict


def check_file_exists(file_path: str) -> None:
    """
    Check if a file exists, raise exception if not

    Args:
        file_path: Path to the file to check

    Raises:
        FileNotFoundError: When the file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


def extract_sections(gallery_docs_raw_content: str) -> List[Dict[str, str]]:
    """
    Extract all sections (category headers and code examples) from GALLERY_DOCS_RAW.md content

    Args:
        gallery_docs_raw_content: Content of GALLERY_DOCS_RAW.md file

    Returns:
        List of dictionaries containing section information:
        - 'type': 'category' or 'example'
        - 'content': section header or (description, code_path) tuple
    """
    sections = []
    lines = gallery_docs_raw_content.split('\n')

    current_description = []

    for line in lines:
        # Check if it's a category header (### N. Category Name)
        category_match = re.match(r'^###\s+\d+\.\s+.+', line)
        if category_match:
            # Save previous example if exists
            if current_description:
                desc_text = '\n'.join(current_description).strip()
                if desc_text:
                    sections.append({
                        'type': 'description',
                        'content': desc_text
                    })
                current_description = []

            # Add category header
            sections.append({
                'type': 'category',
                'content': line
            })
            continue

        # Check if it's a file header (**filename.py** - description)
        file_match = re.match(r'^\*\*(.+?)\*\*\s*-\s*(.+)', line)
        if file_match:
            # Save previous description
            if current_description:
                desc_text = '\n'.join(current_description).strip()
                if desc_text:
                    sections.append({
                        'type': 'description',
                        'content': desc_text
                    })
                current_description = []

            # Add file header
            sections.append({
                'type': 'file_header',
                'content': line
            })
            continue

        # Check if it's a code path tag (at start of line or embedded in line)
        code_path_match = re.search(r'<code_path>(.+?)</code_path>', line)
        if code_path_match:
            # Extract description from the line before the code path tag
            line_before_tag = line[:code_path_match.start()].strip()

            # Add accumulated description
            if current_description:
                desc_text = '\n'.join(current_description).strip()
                if line_before_tag:
                    desc_text = desc_text + ' ' + line_before_tag
            elif line_before_tag:
                desc_text = line_before_tag
            else:
                desc_text = ''

            sections.append({
                'type': 'example',
                'description': desc_text,
                'code_path': code_path_match.group(1)
            })
            current_description = []
            continue

        # Accumulate description text
        if line.strip() and not line.startswith('<!--'):
            current_description.append(line)

    # Save last description if exists
    if current_description:
        desc_text = '\n'.join(current_description).strip()
        if desc_text:
            sections.append({
                'type': 'description',
                'content': desc_text
            })

    return sections


def read_python_file(file_path: str, base_dir: str) -> str:
    """
    Read Python file content

    Args:
        file_path: Relative path to the Python file
        base_dir: Base directory path

    Returns:
        Python file content as string

    Raises:
        FileNotFoundError: When the file does not exist
    """
    full_path = os.path.join(base_dir, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Python file not found: {full_path}")

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content


def generate_template_content(
    gallery_docs_raw_file_path: str,
    base_dir: str
) -> str:
    """
    Generate template content, formatting all code examples as markdown with category headers and table of contents

    Args:
        gallery_docs_raw_file_path: Path to GALLERY_DOCS_RAW.md file
        base_dir: Base directory path

    Returns:
        Formatted markdown content
    """
    with open(gallery_docs_raw_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract all sections (categories and examples)
    sections = extract_sections(content)

    # Generate table of contents from category headers
    toc_entries = []
    for section in sections:
        if section['type'] == 'category':
            # Extract category title for TOC link
            category_line = section['content']
            # Create a simple anchor link from the category name
            toc_entries.append(f"- {category_line.strip()}\n")

    # Generate template content
    template_parts = []

    # Add table of contents
    if toc_entries:
        template_parts.append("\n### 目录索引\n\n")
        template_parts.extend(toc_entries)
        template_parts.append("\n")

    for section in sections:
        section_type = section['type']

        if section_type == 'category':
            # Add category header
            template_parts.append(f"\n{section['content']}\n\n")

        elif section_type == 'file_header':
            # Add file header
            template_parts.append(f"{section['content']}\n\n")

        elif section_type == 'example':
            # Process code example
            description = section['description']
            code_path = section['code_path']

            try:
                python_code = read_python_file(code_path, base_dir)

                # Format as markdown
                formatted_section = f"{description}\n\n```python\n{python_code}\n```\n\n"
                template_parts.append(formatted_section)

            except FileNotFoundError as e:
                print(f"Warning: {e}")
                # Keep description even if file doesn't exist
                formatted_section = f"{description}\n\n```python\n# Code file not found: {code_path}\n```\n\n"
                template_parts.append(formatted_section)

    return ''.join(template_parts)


def generate_docs(
    gallery_all_file_path: str,
    gallery_docs_raw_file_path: str,
    output_file_path: str,
    base_dir: str,
) -> None:
    """
    Generate complete documentation

    Read GALLERY_ALL.md template, replace {template} placeholder with
    code example content generated from GALLERY_DOCS_RAW.md, and output to specified file

    Args:
        gallery_all_file_path: Path to GALLERY_ALL.md template file
        gallery_docs_raw_file_path: Path to GALLERY_DOCS_RAW.md file
        output_file_path: Output file path
        base_dir: Base directory path

    Raises:
        FileNotFoundError: When required files do not exist
    """
    # Check all input files exist
    check_file_exists(gallery_all_file_path)
    check_file_exists(gallery_docs_raw_file_path)

    # Read template file
    with open(gallery_all_file_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Generate code examples content
    code_examples_content = generate_template_content(
        gallery_docs_raw_file_path,
        base_dir
    )

    # Replace {template} placeholder in template
    final_content = template_content.replace(
        '{template}',
        code_examples_content
    )

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write to output file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"Documentation generated successfully: {output_file_path}")
    print(f"Processed {code_examples_content.count('```python')} code examples")


if __name__ == "__main__":
    # Set file paths
    gallery_all_file_path = "codebase/draw/prompts/GALLERY_ALL.md"
    gallery_docs_raw_file_path = "codebase/draw/prompts/GALLERY_DOCS_RAW.md"
    output_file_path = "codebase/draw/README4AGENTS.md"
    base_dir = "./"

    # Generate documentation
    generate_docs(
        gallery_all_file_path=gallery_all_file_path,
        gallery_docs_raw_file_path=gallery_docs_raw_file_path,
        output_file_path=output_file_path,
        base_dir=base_dir,
    )
