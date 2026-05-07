"""Server utilities for ML-Master

This module provides functions for interacting with external validation servers.
"""

import logging
import os
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Default server URLs (configurable via environment)
SERVER_URL_LIST = [
    os.environ.get("ML_MASTER_VALIDATE_SERVER", "http://127.0.0.1:5001"),
]


def is_server_online(max_retries: int = 3, timeout: int = 60) -> tuple[bool, str]:
    """Check if the validation server is online.

    Args:
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_online, server_url)
    """
    import random

    retry = 0
    index = random.randrange(len(SERVER_URL_LIST))
    server_url = SERVER_URL_LIST[index]

    while retry < max_retries:
        try:
            response = requests.get(f"{server_url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Server {server_url} is online, status code: {response.status_code}")
                return True, server_url
            else:
                logger.warning(f"Server returned non-200 status code: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error(f"Connection to {server_url} timed out.")
            timeout += 20
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Connection to {server_url} failed: {e}")

        retry += 1
        if retry < max_retries:
            index = (index + 1) % len(SERVER_URL_LIST)
            server_url = SERVER_URL_LIST[index]
            logger.info(f"Retrying... ({retry}/{max_retries})")
            import time
            time.sleep(1)

    logger.error(f"Server is not online after {max_retries} retries.")
    return False, ""


def call_validate(
    exp_id: str,
    submission_path: str | Path,
    timeout: int = 60,
    max_retries: int = 3
) -> tuple[bool, dict | str]:
    """Validate a submission file by calling the validation server.

    Args:
        exp_id: Experiment ID for tracking
        submission_path: Path to the submission file
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (is_valid, response_or_error)
        - If valid: (True, response_dict) with validation results
        - If invalid: (False, details_dict) with error details
        - If error: (False, error_string)
    """
    import time

    online, server_url = is_server_online()
    if not online:
        return False, f"Validation server at {server_url} is not online"

    retry = 0
    submission_path = Path(submission_path)

    while retry < max_retries:
        try:
            with open(submission_path, "rb") as f:
                files = {"file": f}
                response = requests.post(
                    f"{server_url}/validate",
                    files=files,
                    headers={"exp-id": exp_id},
                    timeout=timeout
                )

            logger.info(f"Validation response status: {response.status_code}")

            try:
                response_json = response.json()
            except ValueError:
                return False, f"Server returned invalid JSON: {response.text[:500]}"

            if "error" in response_json:
                logger.error(f"Server returned error: {response.text}")
                return False, response_json.get('details', 'Unknown error')
            else:
                logger.info("Validation successful")
                return True, response_json

        except requests.exceptions.Timeout:
            logger.error(f"Connection to {server_url} timed out.")
            timeout += 20
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {server_url}, connection error.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
        except FileNotFoundError:
            return False, f"Submission file not found: {submission_path}"
        except Exception as e:
            logger.error(f"Validation failed: {e}")

        retry += 1
        if retry < max_retries:
            logger.info(f"Retrying validation... ({retry}/{max_retries})")
            time.sleep(1)

    return False, "Validation failed after max retries"


def validate_submission_format(
    node_id: str,
    submission_path: str | Path,
    check_format: bool = True
) -> tuple[bool, str]:
    """Validate submission format (wrapper for compatibility).

    Args:
        node_id: Node ID for tracking
        submission_path: Path to submission file
        check_format: Whether to perform validation

    Returns:
        Tuple of (is_valid, message)
    """
    if not check_format:
        return True, "Format validation disabled"

    if not Path(submission_path).exists():
        return False, f"Submission file not found: {submission_path}"

    # Call validation server
    is_valid, result = call_validate(exp_id=node_id, submission_path=submission_path)

    if is_valid:
        return True, "Format validation passed"
    else:
        error_msg = result if isinstance(result, str) else str(result.get('details', 'Unknown error'))
        return False, f"Format validation failed: {error_msg}"
