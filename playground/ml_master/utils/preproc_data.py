"""Data preprocessing utilities for ML-Master

This module provides functions for preprocessing raw data files,
including automatic extraction of compressed archives.
"""

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_tar_file(tar_path: Path | str, extract_path: Path | str) -> bool:
    """Extract a tar file (supports .tar, .tar.gz, .tgz, .tar.bz2).

    Args:
        tar_path: Path to the tar file
        extract_path: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    tar_path = Path(tar_path)
    extract_path = Path(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {tar_path} to {extract_path}...")

    try:
        # Try gzip compression first
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            logger.info(f"Successfully extracted {tar_path} (gzip)")
            return True
        except (tarfile.ReadError, OSError):
            # Try plain tar
            try:
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(path=extract_path)
                logger.info(f"Successfully extracted {tar_path} (plain)")
                return True
            except (tarfile.ReadError, OSError) as e:
                logger.error(f"Failed to extract {tar_path}: {e}")
                return False
    except Exception as e:
        logger.error(f"Error extracting {tar_path}: {e}")
        return False


def extract_zip_file(zip_path: Path | str, extract_path: Path | str) -> bool:
    """Extract a zip file.

    Args:
        zip_path: Path to the zip file
        extract_path: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    zip_path = Path(zip_path)
    extract_path = Path(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path} to {extract_path}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Successfully extracted {zip_path}")
        return True
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False


def preprocess_data(
    data_dir: Path | str,
    recursive: bool = True,
    remove_after_extract: bool = False
) -> dict[str, int]:
    """Preprocess data files by extracting compressed archives.

    Args:
        data_dir: Directory containing data files
        recursive: Whether to search subdirectories
        remove_after_extract: Whether to remove archive after extraction

    Returns:
        Dictionary with extraction statistics
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {"extracted": 0, "failed": 0}

    stats = {"extracted": 0, "failed": 0}

    # Find all compressed files
    if recursive:
        files = list(data_dir.rglob("*.tar")) + \
                list(data_dir.rglob("*.tar.gz")) + \
                list(data_dir.rglob("*.tgz")) + \
                list(data_dir.rglob("*.tar.bz2")) + \
                list(data_dir.rglob("*.zip"))
    else:
        files = list(data_dir.glob("*.tar")) + \
                list(data_dir.glob("*.tar.gz")) + \
                list(data_dir.glob("*.tgz")) + \
                list(data_dir.glob("*.tar.bz2")) + \
                list(data_dir.glob("*.zip"))

    logger.info(f"Found {len(files)} compressed file(s) to process")

    for file_path in files:
        suffix = file_path.suffix.lower()
        parent_dir = file_path.parent

        # Determine extraction method based on file extension
        success = False
        if suffix == ".zip":
            success = extract_zip_file(file_path, parent_dir)
        elif ".tar" in str(file_path):
            success = extract_tar_file(file_path, parent_dir)

        if success:
            stats["extracted"] += 1
            # Optionally remove the original archive
            if remove_after_extract:
                try:
                    file_path.unlink()
                    logger.info(f"Removed original archive: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
        else:
            stats["failed"] += 1

    logger.info(f"Preprocessing complete: {stats['extracted']} extracted, {stats['failed']} failed")
    return stats


def create_directory_structure(workspace_dir: Path | str) -> None:
    """Create the standard ML-Master directory structure.

    Args:
        workspace_dir: Path to the workspace directory
    """
    workspace_dir = Path(workspace_dir)

    directories = [
        workspace_dir / "input",
        workspace_dir / "working",
        workspace_dir / "submission",  # All submissions stored here as submission_{node_id}.csv
        workspace_dir / "best_solution",
        workspace_dir / "best_submission",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def verify_directory_structure(workspace_dir: Path | str) -> bool:
    """Verify that the standard directory structure exists.

    Args:
        workspace_dir: Path to the workspace directory

    Returns:
        True if all directories exist, False otherwise
    """
    workspace_dir = Path(workspace_dir)

    required_dirs = [
        workspace_dir / "input",
        workspace_dir / "working",
        workspace_dir / "submission",
    ]

    for directory in required_dirs:
        if not directory.exists():
            logger.warning(f"Required directory missing: {directory}")
            return False

    return True
