import subprocess
import json
import logging
import time
import os
import requests
import sys
from pathlib import Path
from urllib.parse import urlparse
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("operate-submission")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure repository root is importable so we can reuse grading helpers.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from playground.ml_master_warmup.core.utils.grading_server import ensure_grading_server
except Exception:  # pragma: no cover - optional dependency path
    ensure_grading_server = None


def _get_code_file_path(node_id: str, workspace: str) -> Path:
    """Get the path to the code file for a given node.

    The code file is named `code_{node_id}.py` and represents your
    unique Python script for this task.
    """
    return Path(workspace) / Path(f"code_{node_id}.py")


def _get_submission_file_path(node_id: str, workspace: str) -> Path:
    """Get the path to the submission file for a given node.

    The submission file is located at `submission/submission_{node_id}.csv`
    and represents your unique valid submission file.
    """
    return Path(workspace) / Path("submission") / Path(f"submission_{node_id}.csv")


def _resolve_submission_file_path(node_id: str, workspace: str) -> Path:
    """Resolve submission file path with compatibility fallback.

    Preferred order:
    1) submission/submission_{node_id}.csv
    2) submission/submission.csv
    """
    candidate_node = _get_submission_file_path(node_id, workspace)
    if candidate_node.exists():
        return candidate_node
    return Path(workspace) / "submission" / "submission.csv"


def _is_local_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _http_get(url: str, timeout: int):
    if _is_local_url(url):
        with requests.Session() as session:
            # Avoid proxy interference for localhost.
            session.trust_env = False
            return session.get(url, timeout=timeout)
    return requests.get(url, timeout=timeout)


def _http_post(url: str, *, files: dict, headers: dict, timeout: int):
    if _is_local_url(url):
        with requests.Session() as session:
            # Avoid proxy interference for localhost.
            session.trust_env = False
            return session.post(url, files=files, headers=headers, timeout=timeout)
    return requests.post(url, files=files, headers=headers, timeout=timeout)


@mcp.tool()
def write_code(code: str, node_id: str, workspace: str, override: bool = False) -> str:
    """Save your Python code to the designated code file.

    This tool writes your Python code to `code_{node_id}.py` in your workspace.
    This is your ONLY Python script that will be executed.

    **IMPORTANT**: All file operations MUST be performed within your workspace.
    Submissions created by any other means will be considered INVALID.

    Args:
        code: Your complete Python code as a string (can be multi-line)
        node_id: Your unique node identifier (e.g., "node_abc123")
        workspace: Your working directory path (e.g., "/workspace/exp_001/")
        override: If False, warns when overwriting existing code. If True, overwrites silently.

    Returns:
        Success message with line count, or warning message if code exists and override=False.

    Example:
        >>> write_code(
        ...     code="import pandas as pd\\nprint('Hello World')",
        ...     node_id="node_abc123",
        ...     workspace="/workspace/exp_001/"
        ... )
        'Code successfully written into /workspace/exp_001/code_node_abc123.py, total 2 lines!'
    """
    try:
        code_file = _get_code_file_path(node_id, workspace)
        code_file.parent.mkdir(parents=True, exist_ok=True)
        if not override and code_file.exists():
            # default safe mode
            with open(code_file, "r", encoding="utf-8") as f:
                content = f.read()
            if content:
                return f"Warning: current code has content: {content}\n\n\nIf you want to override it, please set override to True"
        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        line_count = len(code.splitlines())
        logger.info(
            f"Code successfully written into {code_file}, total {line_count} lines!"
        )
        return f"Code successfully written into {code_file}, total {line_count} lines!"

    except Exception as e:
        error_msg = f"Error writing files: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def read_code(node_id: str, workspace: str) -> str:
    """Read your saved code file content.

    This tool reads the content of `code_{node_id}.py` from your workspace.

    Args:
        node_id: Your unique node identifier (e.g., "node_abc123")
        workspace: Your working directory path (e.g., "/workspace/exp_001/")

    Returns:
        The content of the code file as a string, or an error message if the file doesn't exist.
    """
    try:
        code_file = _get_code_file_path(node_id, workspace)

        if not code_file.exists():
            return f"Code file does not exist: {code_file}"

        with open(code_file, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"Code read from {code_file}, total {len(content.splitlines())} lines")
        return content

    except Exception as e:
        error_msg = f"Failed to read code: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def fix_code(
    old_string: str,
    new_string: str,
    node_id: str,
    workspace: str,
    replace_all: bool = False,
) -> str:
    """Perform exact string replacement on your code file.

    This tool edits your `code_{node_id}.py` by replacing exact string matches.
    Use this to fix bugs or make incremental improvements to your code.

    **USAGE GUIDELINES**:
    - Always use `read_code` first to see the exact content before editing
    - The `old_string` must match EXACTLY (including indentation and spacing)
    - Provide enough context in `old_string` to make it unique in the file
    - Use `replace_all=True` when renaming variables throughout the file

    Args:
        old_string: The exact string to replace (must match completely)
        new_string: The new string to replace it with
        node_id: Your unique node identifier (e.g., "node_abc123")
        workspace: Your working directory path (e.g., "/workspace/exp_001/")
        replace_all: If True, replaces all occurrences. If False (default), replaces only the first.

    Returns:
        Success message if replacement succeeded, error message otherwise.

    Example:
        >>> fix_code(
        ...     old_string="model = RandomForestClassifier()",
        ...     new_string="model = RandomForestClassifier(n_estimators=100)",
        ...     node_id="node_abc123",
        ...     workspace="/workspace/exp_001/"
        ... )
        'Code fragment successfully replaced'
    """
    try:
        code_file = _get_code_file_path(node_id, workspace)

        if not code_file.exists():
            return f"Code file does not exist: {code_file}"

        with open(code_file, "r", encoding="utf-8") as f:
            content = f.read()

        if old_string not in content:
            return f"Specified code fragment not found: {old_string[:50]}..."

        # Execute replacement (only first match if replace_all is False)
        if replace_all is False:
            new_content = content.replace(old_string, new_string, 1)
        else:
            # replace all
            new_content = content.replace(old_string, new_string, -1)

        with open(code_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.info(f"Code file {code_file} updated")
        return "Code fragment successfully replaced"

    except Exception as e:
        error_msg = f"Failed to replace code: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def run_code(node_id: str, workspace: str, timeout: int = 300) -> str:
    """Execute your saved Python code file.

    This tool runs your `code_{node_id}.py` script and captures the output.
    The script should generate `submission/submission_{node_id}.csv` as its output.

    **IMPORTANT**: After running, use `validate_submission` to verify your submission
    is valid before using `grade_code` to get your score.

    Args:
        node_id: Your unique node identifier (e.g., "node_abc123")
        workspace: Your working directory path (e.g., "/workspace/exp_001/")
        timeout: Maximum execution time in seconds (default: 300)

    Returns:
        JSON string containing:
        - success: True if execution succeeded
        - stdout: Standard output from the script
        - stderr: Standard error from the script
        - exit_code: Process exit code

    Example:
        >>> run_code(node_id="node_abc123", workspace="/workspace/exp_001/")
        '{"success": true, "stdout": "Training complete...\\n", "stderr": "", "exit_code": 0}'
    """
    try:
        code_file = _get_code_file_path(node_id, workspace)

        if not code_file.exists():
            return json.dumps(
                {"success": False, "error": f"Code file does not exist: {code_file}"},
                ensure_ascii=False,
            )

        # Execute code
        result = subprocess.run(
            ["python", str(code_file)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        response = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }

        logger.info(f"Code execution completed, exit code: {result.returncode}")
        return json.dumps(response, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": "Code execution timed out (exceeded 300 seconds)"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@mcp.tool()
def validate_submission(node_id: str, workspace: str) -> str:
    """Validate your submission file via the grading server.

    This tool checks if your `submission/submission_{node_id}.csv` file is valid
    and obtains a preliminary score from the grading server.

    **USAGE FLOW**: After running your code with `run_code`, use this tool to
    validate your submission before calling `grade_code`.

    Args:
        node_id: Your unique node identifier (e.g., "node_abc123")
        workspace: Your working directory path (e.g., "/workspace/exp_001/")

    Returns:
        JSON string containing:
        - success: True if validation succeeded
        - result: Validation result with score information
        - error: Error message if validation failed

    Example:
        >>> validate_submission(node_id="node_abc123", workspace="/workspace/exp_001/")
        '{"success": true, "result": "Score: 0.85..."}'
    """
    try:
        exp_id = os.environ["ML_MASTER_DATA_EXPID"]
        submission_file = _resolve_submission_file_path(node_id, workspace)
        server_urls_str = os.environ["ML_MASTER_GRADING_SERVERS"]
        data_root = os.environ.get("ML_MASTER_DATA_ROOT", "")

        if not submission_file.exists():
            return json.dumps(
                {"success": False, "error": f"Submission file does not exist: {submission_file}"},
                ensure_ascii=False,
            )

        # Parse server_urls (supports comma-separated multiple URLs)
        urls = [url.strip() for url in server_urls_str.split(",") if url.strip()]
        if not urls:
            return json.dumps(
                {"success": False, "error": "ML_MASTER_GRADING_SERVERS is empty"},
                ensure_ascii=False,
            )

        # Best-effort auto-start grading server (same behavior as warmup playground).
        if ensure_grading_server is not None and data_root:
            try:
                started = ensure_grading_server(
                    dataset_root=data_root,
                    server_urls=urls,
                    startup_timeout=30,
                )
                if started and started not in urls:
                    urls.append(started)
            except Exception as e:
                logger.warning(f"Auto-start grading server failed: {e}")

        # Health check
        server_url = None
        for url in urls:
            try:
                resp = _http_get(f"{url}/health", timeout=60)
                if resp.status_code == 200:
                    server_url = url
                    logger.info(f"Grading server online: {server_url}")
                    break
            except requests.RequestException:
                continue

        if not server_url:
            return json.dumps(
                {"success": False, "error": "Grading server unavailable"},
                ensure_ascii=False,
            )

        # Send validation request (consistent with ml_master format)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(submission_file, "rb") as f:
                    files = {"file": f}
                    resp = _http_post(
                        f"{server_url}/validate",
                        files=files,
                        headers={"exp-id": exp_id},
                        timeout=60,
                    )
                data = resp.json()
                if "error" in data:
                    logger.error(f"Grading server error: {data}")
                    return json.dumps(
                        {"success": False, "error": data.get("details", data["error"])},
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "success": True,
                        "is_valid": data.get("is_valid"),
                        "result": data.get("result"),
                    },
                    ensure_ascii=False,
                )
            except requests.Timeout:
                logger.error(
                    f"Grading validation timeout ({server_url}), attempt {attempt+1}/{max_retries}"
                )
            except requests.RequestException as e:
                logger.error(f"Grading validation failed ({server_url}): {e}")
            time.sleep(1)

        return json.dumps(
            {"success": False, "error": "Grading server call failed"}, ensure_ascii=False
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Validation failed: {str(e)}"}, ensure_ascii=False
        )


@mcp.tool()
def grade_code(node_id: str, workspace: str, timeout: int = 300) -> str:
    """Grade your submission using the local grading script.

    This tool executes the local `grade.py` script to evaluate your submission.
    The script is located at `{data_root}/{exp_id}/prepared/grade.py`.

    **IMPORTANT**: Ensure your submission file exists (created by `run_code`)
    and has been validated (using `validate_submission`) before grading.

    Args:
        node_id: Your unique node identifier (e.g., "node_abc123")
        workspace: Your working directory path (e.g., "/workspace/exp_001/")
        timeout: Maximum execution time in seconds (default: 300)

    Returns:
        JSON string containing:
        - success: True if grading succeeded
        - output: Combined stdout and stderr from the grading script
        - returncode: Process exit code
        - error: Error message if grading failed

    Example:
        >>> grade_code(node_id="node_abc123", workspace="/workspace/exp_001/")
        '{"success": true, "output": "Score: 0.85...", "returncode": 0}'
    """
    try:
        data_root = os.environ["ML_MASTER_DATA_ROOT"]
        exp_id = os.environ["ML_MASTER_DATA_EXPID"]
        submission_file = _resolve_submission_file_path(node_id, workspace)

        # Check if submission file exists
        if not submission_file.exists():
            return json.dumps(
                {
                    "success": False,
                    "error": f"Submission file does not exist: {submission_file}, please use `run_code` tool to generate submission files and use `validate_submission` to ensure it is valid!",
                },
                ensure_ascii=False,
            )

        grade_script = Path(data_root) / exp_id / "prepared" / "grade.py"

        if not grade_script.exists():
            raise ValueError("grade.py does not exist!")
        private_dir = Path(data_root) / exp_id / "prepared" / "private"
        gt_file = private_dir / "test.csv"
        if not gt_file.exists():
            raise ValueError("Ground truth file (test.csv) does not exist!")

        result = subprocess.run(
            [
                "python",
                str(grade_script),
                "-g",
                str(gt_file),
                "-s",
                str(submission_file),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout + result.stderr

        return json.dumps(
            {
                "success": result.returncode == 0,
                "output": output,
                "returncode": result.returncode,
            },
            ensure_ascii=False,
        )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {"success": False, "error": "Grading script execution timed out"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Grading failed: {str(e)}"}, ensure_ascii=False
        )


if __name__ == "__main__":
    mcp.run()
