#!/usr/bin/env python3
"""Grade HTTP 服务器 - 处理使用 ground truth 的测试集评分请求

Usage:
    python initialize_grade_port.py --data-root ${DATA_ROOT} --port 7777

API:
    POST /grade
    {
        "exp_id": "leaf-classification",
        "submission_path": "/path/to/submission.csv",
        "timeout": 300
    }

    Response:
    {
        "success": true,
        "stdout": "...",
        "stderr": "...",
        "returncode": 0
    }
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GradeRequest(BaseModel):
    exp_id: str
    submission_path: str
    timeout: int = 300


class GradeResponse(BaseModel):
    success: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None


app = FastAPI(title="Grade Server", version="1.0.0")

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 线程池用于执行 grade.py
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="grade_")


def run_grade_script(
    exp_id: str, submission_path: str, data_root: Path, timeout: int
) -> tuple[bool, str, str, int]:
    """运行 grade.py 脚本

    Returns:
        (success, stdout, stderr, returncode)
    """
    grade_script = data_root / exp_id / "prepared" / "grade.py"
    if not grade_script.exists():
        return False, "", f"grade.py not found for exp_id={exp_id}", -1

    private_dir = data_root / exp_id / "prepared" / "private"
    csv_files = sorted(private_dir.glob("*.csv"))
    if not csv_files:
        return False, "", f"Ground truth file not found in {private_dir}", -1

    gt_file = csv_files[0]
    submission_file = Path(submission_path)
    if not submission_file.exists():
        return False, "", f"Submission file not found: {submission_path}", -1

    cmd = [
        "python",
        str(grade_script),
        "-g",
        str(gt_file),
        "-s",
        str(submission_file),
    ]

    try:
        logger.info(f"Running grade for {exp_id}: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(data_root),
        )

        logger.info(f"Grade completed for {exp_id}: returncode={proc.returncode}")
        return True, proc.stdout, proc.stderr, proc.returncode

    except subprocess.TimeoutExpired:
        logger.error(f"Grade timed out for {exp_id}")
        return False, "", "Grading script execution timed out", -1
    except Exception as e:
        logger.error(f"Grade failed for {exp_id}: {e}")
        return False, "", f"Grading failed: {str(e)}", -1


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "grade-server"}


@app.post("/grade", response_model=GradeResponse)
async def grade_submission(request: GradeRequest):
    """对提交文件进行评分"""
    data_root = Path(os.environ.get("ML_MASTER_DATA_ROOT", "${DATA_ROOT}"))

    loop = asyncio.get_event_loop()
    success, stdout, stderr, returncode = await loop.run_in_executor(
        executor,
        run_grade_script,
        request.exp_id,
        request.submission_path,
        data_root,
        request.timeout,
    )

    if success:
        return GradeResponse(
            success=True,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )
    else:
        return GradeResponse(
            success=False,
            error=stderr,
        )


def main():
    parser = argparse.ArgumentParser(description="Grade HTTP Server")
    parser.add_argument(
        "--data-root",
        default="${DATA_ROOT}",
        help="Path to ML_MASTER_DATA_ROOT"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7777, help="Bind port (default: 7777)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    args = parser.parse_args()

    # 设置环境变量
    os.environ["ML_MASTER_DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())

    logger.info(f"Starting Grade HTTP Server on http://{args.host}:{args.port}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Workers: {args.workers}")

    uvicorn.run(
        "initialize_grade_port:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()