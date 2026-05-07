#!/usr/bin/env python3
"""测试 grading_server 的启动和评测流程。

测试固定 exp_id: detecting-insults-in-social-commentary
测试 files 目录下的三个文件: empty.csv, false.csv, true.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 添加 playground 到 path
playground_root = Path(__file__).parent.parent / "playground" / "ml_master"
sys.path.insert(0, str(playground_root))

from core.utils.grading_server import ensure_grading_server, stop_grading_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 固定配置
EXP_ID = "detecting-insults-in-social-commentary"
DATA_ROOT = "/data/public_data/exp_data/demo1bench"
TEST_FILES_DIR = Path("${PROJECT_ROOT}/test/files")



def validate_submission_via_server(
    server_url: str,
    exp_id: str,
    submission_path: Path,
    timeout: int = 60,
) -> tuple[bool, str | dict]:
    """通过 HTTP 调用 grading server 进行评测。

    模拟真实业务中的 HTTP 请求流程。
    """
    import requests

    try:
        with open(submission_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(
                f"{server_url}/validate",
                files=files,
                headers={"exp-id": exp_id},
                timeout=timeout,
            )

        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text}"

        data = resp.json()
        if "error" in data:
            return False, data.get("details", data["error"])

        return True, data
    except requests.Timeout:
        return False, "timeout"
    except requests.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, f"unexpected error: {e}"


def test_grading_server():
    """测试 grading server 的完整流程。

    测试 files 目录下的三个文件:
    - empty.csv: 空文件
    - false.csv: 格式错误（缺少表头）
    - true.csv: 有表头和数据
    """
    import requests

    # 硬编码测试文件
    test_files = {
        "empty": TEST_FILES_DIR / "empty.csv",
        "false": TEST_FILES_DIR / "false.csv",
        "true": TEST_FILES_DIR / "true.csv",
    }

    # 检查文件是否存在
    for name, path in test_files.items():
        if not path.exists():
            logger.error(f"测试文件不存在: {path}")
            return False
    logger.info(f"测试文件: {list(test_files.keys())}")

    # ============ 阶段1: 启动服务器 ============
    logger.info("=" * 60)
    logger.info(f"阶段1: 启动 grading server (exp_id={EXP_ID})")
    logger.info("=" * 60)

    server_url = ensure_grading_server(
        dataset_root=DATA_ROOT,
        server_urls=["http://127.0.0.1:5003"],
        startup_timeout=30,
    )

    if not server_url:
        logger.error("无法启动 grading server")
        return False

    logger.info(f"Grading server 已启动: {server_url}")

    # 健康检查
    try:
        health_resp = requests.get(f"{server_url}/health", timeout=5)
        if health_resp.status_code == 200:
            logger.info("健康检查通过 ✓")
        else:
            logger.warning(f"健康检查返回: {health_resp.status_code}")
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return False

    # ============ 阶段2: 测试三个文件 ============
    logger.info("")
    logger.info("=" * 60)
    logger.info("阶段2: 测试三个文件的格式校验")
    logger.info("=" * 60)

    results = {}  # name -> (request_success, is_valid, message)
    for name, path in test_files.items():
        logger.info(f"测试文件: {name}")

        request_success, result = validate_submission_via_server(
            server_url=server_url,
            exp_id=EXP_ID,
            submission_path=path,
        )

        if request_success:
            is_valid = result.get("is_valid", False)
            message = result.get("result", "N/A")
            results[name] = (True, is_valid, message)
            status = "✓ VALID" if is_valid else "✗ INVALID"
            logger.info(f"  状态: {status}")
            logger.info(f"  消息: {message}")
        else:
            results[name] = (False, False, str(result))
            logger.error(f"  请求失败: {result}")

    # ============ 阶段3: 关闭服务器 ============
    logger.info("")
    logger.info("=" * 60)
    logger.info("阶段3: 关闭 grading server")
    logger.info("=" * 60)

    stopped = stop_grading_server(timeout=5)
    if stopped:
        logger.info("服务器已关闭 ✓")
    else:
        logger.warning("服务器关闭失败或服务器非本进程启动")

    # ============ 汇总结果 ============
    logger.info("")
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)

    for name, (request_success, is_valid, message) in results.items():
        if request_success:
            status = "✓ VALID" if is_valid else "✗ INVALID"
            logger.info(f"{name:15s}: {status} - {message}")
        else:
            logger.error(f"{name:15s}: 请求失败 - {message}")

    # 测试成功 = 所有 HTTP 请求都成功完成
    return all(request_success for request_success, _, _ in results.values())


if __name__ == "__main__":
    success = test_grading_server()
    sys.exit(0 if success else 1)
