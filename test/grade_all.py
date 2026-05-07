#!/usr/bin/env python3
"""
对 submission 目录中的全部 submission 进行评测
"""

import os
import re
import subprocess
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

SUBMISSION_DIR = "${PROJECT_ROOT}/runs/ml_master_datatree_20260326_165235/workspaces/task_0/submission"
GRADE_SCRIPT = "${DATA_ROOT}/detecting-insults-in-social-commentary/prepared/grade.py"
OUTPUT_FILE = "${PROJECT_ROOT}/test/grade_results.json"


def extract_score(output: str) -> float:
    """从 grade.py 输出中提取分数"""
    # 匹配 pattern: metric = ... 0.78606 (可能包含 ANSI 颜色码)
    match = re.search(r"metric\s*=.*?(\d+\.\d+)", output)
    if match:
        return float(match.group(1))
    return None


def grade_submission(submission_path: str) -> Tuple[str, float]:
    """对单个 submission 进行评测"""
    submission_name = os.path.basename(submission_path)
    print(f"Grading {submission_name}...")

    try:
        result = subprocess.run(
            ["python", GRADE_SCRIPT, "-s", submission_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        score = extract_score(output)
        if score is not None:
            print(f"  Score: {score:.5f}")
        return submission_name, score
    except subprocess.TimeoutExpired:
        return submission_name, None
    except Exception as e:
        return submission_name, None


def main():
    submission_dir = Path(SUBMISSION_DIR)

    if not submission_dir.exists():
        print(f"Error: Submission directory does not exist: {SUBMISSION_DIR}")
        return

    # 获取所有 CSV 文件
    submission_files = list(submission_dir.glob("submission_*.csv"))
    print(f"Found {len(submission_files)} submission files")

    if not submission_files:
        print("No submission files found!")
        return

    # 多线程评测
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(grade_submission, str(f)): str(f) for f in submission_files
        }

        for future in as_completed(futures):
            submission_name, score = future.result()
            # 提取 id: 从 submission_6374c7938eb243488c7151932c3f3d40.csv -> 6374c7938eb243488c7151932c3f3d40
            submission_id = submission_name.replace("submission_", "").replace(
                ".csv", ""
            )
            results[submission_id] = {"score": score}

    # 保存结果
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # 统计
    scores = [r["score"] for r in results.values() if r["score"] is not None]
    if scores:
        print("\n" + "=" * 60)
        print(f"Total submissions: {len(submission_files)}")
        print(f"Successfully graded: {len(scores)}")
        print(f"Failed: {len(submission_files) - len(scores)}")
        print(f"\nScore statistics:")
        print(f"  Max: {max(scores):.5f}")
        print(f"  Min: {min(scores):.5f}")
        print(f"  Mean: {sum(scores)/len(scores):.5f}")
        print(f"  Median: {sorted(scores)[len(scores)//2]:.5f}")
        print("=" * 60)
        print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
