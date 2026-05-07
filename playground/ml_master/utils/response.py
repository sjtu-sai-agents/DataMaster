"""Response parsing utilities for ML-Master"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_code(text: str) -> str | None:
    """Extract code from markdown code blocks.

    Args:
        text: The text to extract code from

    Returns:
        The extracted code, or None if no code blocks found
    """
    # Match both ```python and ``` code blocks
    pattern = r'```(?:python|py)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Join multiple code blocks with newlines
        return '\n'.join(matches)

    # Try to extract any code-like content
    if 'def ' in text or 'import ' in text or 'class ' in text:
        # Find code-like section
        lines = text.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                in_code = not in_code
                continue
            if in_code or any(kw in line for kw in ['import ', 'from ', 'def ', 'class ', 'print(']):
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

    return None


def extract_text_up_to_code(text: str) -> str:
    """Extract text before the first code block.

    Args:
        text: The text to extract from

    Returns:
        The text before the first code block
    """
    # Find the first code block
    match = re.search(r'```', text)
    if match:
        return text[:match.start()].strip()
    return text.strip()


def wrap_code(code: str, lang: str = "python") -> str:
    """Wrap code in a markdown code block.

    Args:
        code: The code to wrap
        lang: The language identifier

    Returns:
        The wrapped code
    """
    return f"```{lang}\n{code}\n```"


def extract_review(text: str) -> dict | None:
    """Extract JSON review from text.

    Args:
        text: The text containing the JSON review

    Returns:
        The parsed review dict, or None if parsing fails
    """
    # Try to find JSON code block
    pattern = r'```json\s*\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        json_str = match.group(1)
    else:
        # Try to find any JSON-like content
        json_start = text.find('{')
        json_end = text.rfind('}')
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end + 1]
        else:
            return None

    try:
        import json
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON review: {e}")
        return None


def trim_long_string(s: str, max_length: int = 10000) -> str:
    """Trim a string to a maximum length.

    Args:
        s: The string to trim
        max_length: Maximum length

    Returns:
        The trimmed string with ellipsis if needed
    """
    if len(s) <= max_length:
        return s
    return s[:max_length] + "... (truncated)"


def extract_metric_from_output(output: str) -> float | None:
    """Extract a metric value from terminal output.

    Looks for patterns like:
    - "Validation AUC: 0.8542"
    - "accuracy = 0.9234"
    - "Score: 0.78"

    Args:
        output: The terminal output to search

    Returns:
        The extracted metric value, or None if not found
    """
    # Common metric patterns
    patterns = [
        r'(?:validation|val|score|accuracy|auc|rmse|mae|f1)[-:\s]+([0-9]+\.?[0-9]*)',
        r'([0-9]+\.[0-9]+)(?:\s*(?:validation|val|score|accuracy))',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])  # Use last match
            except ValueError:
                continue

    return None
