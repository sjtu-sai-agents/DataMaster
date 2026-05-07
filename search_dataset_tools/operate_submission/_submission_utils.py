"""Shared utilities and grading server for operate_submission tools."""

import sys
import asyncio
import subprocess
import json
import logging
import time
import os
import requests
import shlex
import re
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import urlparse

import signal
import socket
import threading
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from mlebench.grade import validate_submission as _bench_validate
from mlebench.registry import registry

# ========================================================================
# 资源管理全局锁
# ========================================================================

_CPU_SELECT_LOCK = threading.Lock()
_GPU_SELECT_LOCK = threading.Lock()


# ========================================================================
# CPU 自动选择
# ========================================================================


# 缓存上一次的 CPU 统计数据，用于计算占用率
_cpu_stats_cache: dict[str, list[int]] | None = None
_cpu_stats_time: float = 0


def _read_cpu_stats() -> dict[str, list[int]]:
    """读取 /proc/stat 获取 CPU 核心统计信息。

    Returns:
        字典，key 为 cpu_id (如 "0", "1", ...)，value 为 [user, nice, system, idle, iowait, irq, softirq]
    """
    stats = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3:4].isdigit():  # cpu0, cpu1, ..., 排除 cpu
                parts = line.split()
                cpu_id = parts[0][3:]  # "cpu0" -> "0"
                # user, nice, system, idle, iowait, irq, softirq, steal
                values = [int(x) for x in parts[1:8]]
                stats[cpu_id] = values
    return stats


def _get_cpu_info() -> list[dict] | None:
    """获取所有 CPU 核心的占用率。

    通过两次采样 /proc/stat 计算每个核心的占用率。

    Returns:
        包含每个 CPU 核心信息的列表，每个元素包含:
        - index: CPU 核心编号
        - utilization: CPU 利用率百分比 (0-100)
        如果查询失败则返回 None
    """
    global _cpu_stats_cache, _cpu_stats_time

    try:
        # 第一次采样
        stats1 = _read_cpu_stats()
        time.sleep(0.1)  # 短暂等待
        # 第二次采样
        stats2 = _read_cpu_stats()

        cpus = []
        for cpu_id in sorted(stats1.keys(), key=int):
            v1 = stats1[cpu_id]
            v2 = stats2.get(cpu_id, v1)

            # 计算差值
            delta = [v2[i] - v1[i] for i in range(len(v1))]
            total_delta = sum(delta)

            if total_delta == 0:
                utilization = 0
            else:
                idle_delta = delta[3]  # idle is at index 3
                utilization = 100 * (total_delta - idle_delta) / total_delta

            cpus.append({
                "index": int(cpu_id),
                "utilization": utilization,
            })

        return cpus if cpus else None

    except Exception as e:
        logger.warning(f"Failed to get CPU info: {e}")
        return None


def _select_best_cpus(n_cores: int = 32) -> list[int] | None:
    """选择占用率最低的 N 个 CPU 核心。

    使用全局锁避免并发竞态。

    Args:
        n_cores: 需要选择的 CPU 核心数量，默认 32

    Returns:
        选择的 CPU 核心编号列表，如果没有 CPU 则返回 None
    """
    with _CPU_SELECT_LOCK:
        cpus = _get_cpu_info()
        if not cpus:
            return None

        # 按利用率排序，选择最低的 N 个
        cpus.sort(key=lambda x: x["utilization"])
        selected = cpus[:n_cores]

        cpu_ids = [cpu["index"] for cpu in selected]
        avg_util = sum(cpu["utilization"] for cpu in selected) / len(selected)

        logger.info(
            f"Selected {len(cpu_ids)} CPUs: {cpu_ids[:4]}...{cpu_ids[-4:]}, "
            f"avg_util={avg_util:.1f}%"
        )

        return cpu_ids


def _format_cpu_list_for_taskset(cpu_ids: list[int]) -> str:
    """将 CPU ID 列表格式化为 taskset 使用的掩码格式。

    Args:
        cpu_ids: CPU 核心编号列表

    Returns:
        taskset -c 参数格式的字符串，如 "0-31,64,65"
    """
    if not cpu_ids:
        return ""

    # 先排序
    sorted_ids = sorted(cpu_ids)

    # 尝试合并连续的范围
    ranges = []
    start = sorted_ids[0]
    prev = start

    for cpu_id in sorted_ids[1:]:
        if cpu_id == prev + 1:
            prev = cpu_id
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = cpu_id
            prev = cpu_id

    # 处理最后一个范围
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")

    return ",".join(ranges)


# ========================================================================
# GPU 自动选择
# ========================================================================


def _get_tmux_gpu_limit() -> int | None:
    """检查 tmux session 名称，推断 GPU 数量限制。

    新逻辑：
    - 如果存在名字匹配 '(\\d+)_gpu' 的 tmux session，例如:
      - 4_gpu
      - train_4_gpu
      - ablation_8_gpu
      则返回对应数值。若有多个，取最小值，更保守。

    兼容旧逻辑：
    - 如果存在包含 'flag' 的会话，则返回 5。

    Returns:
        GPU 数量限制；若未检测到限制则返回 None。
    """
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", r"#{session_name}"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            # tmux server 可能未运行或不在 tmux 环境中
            return None

        session_names = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ]

        gpu_limits = []
        for name in session_names:
            m = re.search(r"(\d+)_gpu\b", name.lower())
            if m:
                gpu_limits.append(int(m.group(1)))

        if gpu_limits:
            limit = min(gpu_limits)
            logger.info(
                f"Detected tmux GPU limit from session names {session_names}: {limit}"
            )
            return limit

        # 兼容旧逻辑
        for name in session_names:
            if "flag" in name.lower():
                logger.info(f"Found legacy tmux session with 'flag': {name}, limit=5")
                return 5

        return None

    except FileNotFoundError:
        # tmux 未安装
        return None
    except subprocess.TimeoutExpired:
        logger.warning("tmux list-sessions timeout")
        return None
    except Exception as e:
        logger.warning(f"Failed to check tmux sessions: {e}")
        return None


def _get_gpu_info() -> list[dict] | None:
    """获取所有 GPU 的信息，包括 util 和显存占用。

    动态 GPU 资源池策略:
    - 如果 tmux 中存在 '(N)_gpu' 的会话名，只使用前 N 个 GPU
      例如 '4_gpu' -> 只使用前 4 个 GPU
    - 否则兼容旧逻辑：若存在包含 'flag' 的会话，只使用前 5 个 GPU
    - 否则，使用全部 GPU

    Returns:
        包含每个 GPU 信息的列表，每个元素包含:
        - index: GPU 编号
        - utilization_gpu: GPU 利用率百分比
        - memory_used: 已用显存 (MB)
        - memory_total: 总显存 (MB)
        - memory_utilization: 显存利用率百分比
        如果没有 GPU 或查询失败则返回 None
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed: {result.stderr}")
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) != 4:
                continue
            index = int(parts[0].strip())
            util_gpu = int(parts[1].strip())
            mem_used = int(parts[2].strip())
            mem_total = int(parts[3].strip())
            mem_util = (mem_used / mem_total * 100) if mem_total > 0 else 0

            gpus.append({
                "index": index,
                "utilization_gpu": util_gpu,
                "memory_used": mem_used,
                "memory_total": mem_total,
                "memory_utilization": mem_util,
            })

        if not gpus:
            return None

        gpu_limit = _get_tmux_gpu_limit()

        if gpu_limit is not None:
            original_count = len(gpus)
            gpus = gpus[:gpu_limit]
            logger.info(
                f"TMUX GPU limit detected, limiting GPU pool from {original_count} "
                f"to first {len(gpus)} GPUs: {[gpu['index'] for gpu in gpus]}"
            )
        else:
            logger.info(
                f"No TMUX GPU limit detected, using all {len(gpus)} GPUs: "
                f"{[gpu['index'] for gpu in gpus]}"
            )

        return gpus

    except FileNotFoundError:
        logger.warning("nvidia-smi not found, no GPU available")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timeout")
        return None
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
        return None


def _select_best_gpu() -> int | None:
    """选择显存占用最少的 GPU。

    使用全局锁避免并发竞态。

    Returns:
        选择的 GPU 编号，如果没有 GPU 则返回 None
    """
    with _GPU_SELECT_LOCK:
        gpus = _get_gpu_info()
        if not gpus:
            return None

        # 按显存利用率排序，选择最低的
        gpus.sort(key=lambda x: x["memory_utilization"])
        best = gpus[0]

        logger.info(
            f"Selected GPU {best['index']}: "
            f"mem_used={best['memory_used']}MB/{best['memory_total']}MB "
            f"({best['memory_utilization']:.1f}%)"
        )

        return best["index"]


logger = logging.getLogger(__name__)

_SERVER_LOCK = threading.Lock()
_SERVER_THREAD: threading.Thread | None = None
_SERVER_HTTPD = None
_SERVER_PROC: subprocess.Popen | None = None
_SERVER_URL: str | None = None
_SERVER_OWNED: bool = False


# ========================================================================
# Grading Server
# ========================================================================


def _create_app(base_dir: Path) -> Flask:
    """Create Flask app bound to a dataset directory."""
    app = Flask(__name__)
    private_dir = Path(base_dir)
    new_registry = registry.set_data_dir(private_dir)

    def run_validation(submission: Path, competition_id: str) -> Tuple[bool, str]:
        competition = new_registry.get_competition(competition_id)
        is_valid, message = _bench_validate(submission, competition)
        return is_valid, message

    @app.route("/validate", methods=["POST"])
    def validate():
        submission_file = request.files["file"]
        competition_id = request.headers.get("exp-id")
        submission_path = Path("/tmp/submission_to_validate.csv")
        submission_file.save(submission_path)

        try:
            is_valid, result = run_validation(submission_path, competition_id)
        except Exception as exc:  # noqa: BLE001
            return (
                jsonify(
                    {"error": "An unexpected error occurred.", "details": str(exc)}
                ),
                500,
            )

        return jsonify({"result": result, "is_valid": is_valid})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "running"}), 200

    return app


def _is_local_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _http_get(url: str, timeout: int):
    if _is_local_url(url):
        with requests.Session() as session:
            # Keep global proxies for external requests, but bypass for localhost.
            session.trust_env = False
            return session.get(url, timeout=timeout)
    return requests.get(url, timeout=timeout)


def _parse_host_port(
    url: str,
    default_host: str = "127.0.0.1",
    default_port: int | None = None,
) -> Tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or default_host
    port = parsed.port or default_port
    return host, port


def _is_healthy(url: str, timeout: int = 5) -> bool:
    try:
        resp = _http_get(url.rstrip("/") + "/health", timeout=timeout)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _wait_for_health(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_healthy(url, timeout=2):
            return True
        time.sleep(0.5)
    return False


def _is_local_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _is_port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check whether TCP port is already occupied."""
    probe_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((probe_host, port)) == 0
        except OSError:
            return False


def _pick_bind_target(urls: list[str], default_port: int | None = None) -> Tuple[str, int]:
    """Choose a local host/port to start embedded grading server.

    Strategy:
    - Prefer candidate urls in order.
    - Skip candidates whose local port is occupied.
    - If all candidates are occupied, scan localhost ports from default_port upward.
    """
    # If no default port specified, try to extract from first URL
    if default_port is None and urls:
        first_host, first_port = _parse_host_port(urls[0], default_port=default_port)
        default_port = first_port
    if default_port is None:
        default_port = 5003  # Fallback default

    candidates: list[Tuple[str, int]] = []
    for url in urls:
        host, port = _parse_host_port(url, default_port=default_port)
        if not _is_local_host(host):
            continue
        candidates.append((host, port))

    if not candidates:
        candidates.append(("127.0.0.1", default_port))

    for host, port in candidates:
        if not _is_port_in_use(host, port):
            return host, port

    scan_host = "127.0.0.1"
    for port in range(default_port, default_port + 200):
        if not _is_port_in_use(scan_host, port):
            return scan_host, port

    return "127.0.0.1", default_port


def ensure_grading_server(
    dataset_root: str | Path | None,
    server_urls: Iterable[str] | None = None,
    startup_timeout: int = 30,
) -> str:
    """Ensure a grading server endpoint is available on the configured fixed port.

    逻辑：
    - 只使用传入/环境变量中的唯一固定 URL
    - 如果健康，直接复用
    - 如果不健康，则在同一个 host:port 上启动/重启 standalone grading server
    """
    global _SERVER_URL, _SERVER_OWNED, _SERVER_PROC  # noqa: PLW0603

    with _SERVER_LOCK:
        urls = [u.strip() for u in (server_urls or []) if str(u).strip()]
        if not urls:
            default_servers = os.environ.get("ML_MASTER_GRADING_SERVERS", "")
            if default_servers:
                urls = [u.strip() for u in default_servers.split(",") if u.strip()]

        if not urls:
            logger.warning("No grading server url configured")
            return ""

        if len(urls) != 1:
            logger.warning("Expected exactly one grading server url, got: %s", urls)

        target_url = urls[0]

        if _is_healthy(target_url):
            _SERVER_URL = target_url
            _SERVER_OWNED = False
            logger.info("Using existing healthy grading server: %s", target_url)
            return target_url

        if not dataset_root:
            logger.warning(
                "Grading server unhealthy and dataset_root is missing; cannot restart."
            )
            return ""

        # 如果当前进程以前起过旧 server，先停掉
        if _SERVER_PROC is not None:
            try:
                _SERVER_PROC.terminate()
                _SERVER_PROC.wait(timeout=5)
            except Exception:
                try:
                    _SERVER_PROC.kill()
                except Exception:
                    pass
            finally:
                _SERVER_PROC = None

        host, port = _parse_host_port(target_url, default_port=None)
        if port is None:
            logger.warning("Configured grading server url missing port: %s", target_url)
            return ""

        project_root = Path(__file__).resolve().parents[2]
        runner_path = (
            project_root
            / "search_dataset_tools"
            / "operate_submission"
            / "grading_server_runner.py"
        )

        cmd = [
            sys.executable,
            str(runner_path),
            "--data-root",
            str(dataset_root),
            "--host",
            host,
            "--port",
            str(port),
        ]

        logger.warning(
            "Restarting grading server on same port: %s (dataset_root=%s)",
            target_url,
            dataset_root,
        )

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

        if _wait_for_health(target_url, timeout=startup_timeout):
            _SERVER_PROC = proc
            _SERVER_URL = target_url
            _SERVER_OWNED = True
            logger.info("Standalone grading server ready on %s", target_url)
            return target_url

        try:
            proc.terminate()
        except Exception:
            pass

        logger.warning("Failed to make grading server healthy on %s", target_url)
        return ""


def stop_grading_server(timeout: int = 5) -> bool:
    """Stop standalone grading server started by current process.

    Returns True when a local owned process was stopped.
    """
    global _SERVER_THREAD, _SERVER_HTTPD, _SERVER_PROC, _SERVER_URL, _SERVER_OWNED  # noqa: PLW0603

    with _SERVER_LOCK:
        stopped = False

        if _SERVER_PROC is not None:
            try:
                _SERVER_PROC.terminate()
                _SERVER_PROC.wait(timeout=timeout)
                stopped = True
                logger.info("Standalone grading server stopped")
            except Exception as exc:
                logger.warning("Failed to stop grading server process cleanly: %s", exc)
                try:
                    _SERVER_PROC.kill()
                except Exception:
                    pass

        _SERVER_THREAD = None
        _SERVER_HTTPD = None
        _SERVER_PROC = None
        _SERVER_URL = None
        _SERVER_OWNED = False
        return stopped


# ========================================================================
# Helper Functions
# ========================================================================


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


def _get_template_file_path(node_id: str, workspace: str) -> Path:
    """Get the path to the template file for a given node.

    The template file is named `code_{node_id}_template.py` and contains
    fixed code (BaseDataLoader + training framework).
    """
    return Path(workspace) / Path(f"code_{node_id}_template.py")


def _get_dataloader_file_path(node_id: str, workspace: str) -> Path:
    """Get the path to the dataloader file for a given node.

    The dataloader file is named `code_{node_id}_dataloader.py` and contains
    only the MyDataLoader derived class implementation.
    """
    return Path(workspace) / Path(f"code_{node_id}_dataloader.py")


def _get_base_dataloader_path() -> Path:
    """Get the path to the base dataloader file.

    The base dataloader file contains the BaseDataLoader abstract class
    and is located at search_dataset_tools/operate_submission/base_dataloader.py.
    """
    return Path(__file__).resolve().parent / "base_dataloader.py"
    


def _is_separated_mode(node_id: str, workspace: str) -> bool:
    """Check if the node uses separated code structure.

    Returns True if both template and dataloader files exist.
    """
    template_path = _get_template_file_path(node_id, workspace)
    dataloader_path = _get_dataloader_file_path(node_id, workspace)
    return template_path.exists() and dataloader_path.exists()


def _assemble_code(node_id: str, workspace: str) -> str | None:
    """Assemble base_dataloader + dataloader + template into complete executable code.

    Assembly order (as specified in README):
    1. base_dataloader code (BaseDataLoader abstract class)
    2. "\n\n"
    3. MyDataLoader class definition (dataloader file)
    4. template's run script (training code)

    Returns:
        Assembled code string, or None if assembly fails.
    """
    base_dataloader_path = _get_base_dataloader_path()
    dataloader_path = _get_dataloader_file_path(node_id, workspace)
    template_path = _get_template_file_path(node_id, workspace)

    if not base_dataloader_path.exists():
        logger.error(f"Base dataloader file not found: {base_dataloader_path}")
        return None
    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        return None
    if not dataloader_path.exists():
        logger.error(f"Dataloader file not found: {dataloader_path}")
        return None

    try:
        base_dataloader_content = base_dataloader_path.read_text(encoding="utf-8")
        dataloader_content = dataloader_path.read_text(encoding="utf-8")
        template_content = template_path.read_text(encoding="utf-8")

        # Assembly order: base_dataloader + "\n\n" + MyDataLoader + template
        separator1 = "\n\n"
        separator2 = "\n\n"

        assembled = (
            base_dataloader_content
            + separator1
            + dataloader_content
            + separator2
            + template_content
        )
        logger.info(f"Successfully assembled code for node {node_id}")
        return assembled

    except Exception as e:
        logger.error(f"Assembly error for node {node_id}: {e}")
        return None


def _load_resources(workspace: str) -> dict:
    """从 workspace/.resources.json 读取 GPU/CPU 资源分配信息。

    该文件由 execute_parallel_tasks 在每个并行槽启动时自动写入，
    记录当前 worker 被分配到的 CUDA_VISIBLE_DEVICES 和 CPU affinity。
    """
    resource_file = Path(workspace) / ".resources.json"
    if not resource_file.exists():
        return {}
    try:
        with open(resource_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read resource file {resource_file}: {e}")
        return {}


def _auto_replace_submission_file(
    node_id: str, workspace: str
) -> tuple[bool, str | None]:
    """自动重命名 submission.csv 为 submission_{node_id}.csv。

    直接检查文件系统，如果 workspace/submission/submission.csv 存在，
    则自动重命名为 submission_{node_id}.csv。

    Args:
        node_id: 节点 ID
        workspace: 工作目录路径

    Returns:
        (是否成功替换, 替换信息消息)
    """
    workspace_path = Path(workspace)

    # 直接检查 submission/submission.csv 文件是否存在
    source_file = workspace_path / "submission" / "submission.csv"
    if not source_file.exists():
        return False, None

    # 重命名为 submission_{node_id}.csv
    target_submission_file = _get_submission_file_path(node_id, workspace)
    target_submission_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import shutil
        shutil.move(str(source_file), str(target_submission_file))
        message = f"Auto-renamed submission.csv -> submission_{node_id}.csv"
        logger.info(f"[{node_id}] {message}")
        return True, message
    except Exception as e:
        message = f"Failed to rename submission.csv: {str(e)}"
        logger.error(f"[{node_id}] {message}")
        return False, message


# ========================================================================
# Shared async tool functions (to reduce code duplication)
# ========================================================================


async def _run_code_async(
    node_id: str, workspace: str, timeout: int, load_resources_fn, logger
) -> str:
    """Shared implementation for run_code tool.

    资源管理:
    - CPU: 自动选择占用最低的 32 个核心，使用 taskset 绑定
    - GPU: 自动选择 util 和显存占用最少的 GPU，设置 CUDA_VISIBLE_DEVICES
    """
    try:
        code_file = _get_code_file_path(node_id, workspace)

        # Check if separated mode and assemble code if needed
        if _is_separated_mode(node_id, workspace):
            logger.info(
                f"Detected separated mode for node {node_id}, assembling code..."
            )
            assembled_code = _assemble_code(node_id, workspace)
            if assembled_code is None:
                return json.dumps(
                    {"success": False, "error": "Code assembly failed"},
                    ensure_ascii=False,
                )
            code_file.parent.mkdir(parents=True, exist_ok=True)
            code_file.write_text(assembled_code, encoding="utf-8")
            logger.info(f"Assembled code written to {code_file}")

        if not code_file.exists():
            return json.dumps(
                {"success": False, "error": f"Code file does not exist: {code_file}"},
                ensure_ascii=False,
            )

        env = os.environ.copy()

        # GPU 选择逻辑：
        # 1. 如果外部已经显式设置 CUDA_VISIBLE_DEVICES，则优先尊重外部设置
        # 2. 否则再走自动选卡逻辑
        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_device:
            env["CUDA_VISIBLE_DEVICES"] = cuda_device
            logger.info(
                f"run_code: Using CUDA_VISIBLE_DEVICES from env: {cuda_device}"
            )
        else:
            best_gpu = _select_best_gpu()
            if best_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                logger.info(f"run_code: Auto-selected GPU {best_gpu}")

        # 自动选择最优 CPU 核心（默认 32 个）
        cpu_cores = _select_best_cpus(n_cores=32)

        start_time = time.time()

        # 构建命令：如果选择了 CPU，使用 taskset 包装
        if cpu_cores:
            cpu_list_str = _format_cpu_list_for_taskset(cpu_cores)
            cmd = [
                "taskset",
                "-c", cpu_list_str,
                "python",
                str(code_file),
            ]
            logger.info(f"run_code: Using taskset -c {cpu_list_str}")
        else:
            cmd = [
                "python",
                str(code_file),
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
            env=env,
            start_new_session=True,  # 创建新进程组，便于 killpg 杀死整个进程树
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            # 杀死整个进程组（包括 taskset 和 python 子进程）
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass  # 进程已经结束
            await proc.wait()
            return json.dumps(
                {
                    "success": False,
                    "error": f"Code execution timed out (exceeded {timeout} seconds)",
                },
                ensure_ascii=False,
            )

        elapsed_time = time.time() - start_time
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Auto-replace submission file if exists
        submission_replaced, replacement_message = _auto_replace_submission_file(
            node_id, workspace
        )

        response = {
            "success": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
            "elapsed_time": elapsed_time,
        }

        # Add submission replacement info to response
        if replacement_message is not None:
            response["submission_replaced"] = submission_replaced
            response["replacement_message"] = replacement_message

        logger.info(
            f"Code execution completed, using time: {elapsed_time:.2f}s\nexit code: {proc.returncode}"
        )

        # Save execution result to cache
        cache_dir = Path(workspace) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{node_id}.json"

        cache_data = {
            "stdout": response.get("stdout", ""),
            "exit_code": response.get("exit_code", -1),
            "stderr": response.get("stderr", ""),
            "success": response.get("success", False),
            "elapsed_time": response.get("elapsed_time", 0),
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved execution cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

        return json.dumps(response, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


async def _validate_submission_async(
    node_id: str, workspace: str, ensure_server_fn, logger
) -> str:
    """Shared implementation for validate_submission tool.

    修复点：
    - 只使用固定的 config grading server url
    - validate 前先探活
    - 不健康时在同一个端口重启
    - server unavailable 和 invalid submission 严格区分
    """
    try:
        exp_id = os.environ["ML_MASTER_DATA_EXPID"]
        data_root = os.environ.get("ML_MASTER_DATA_ROOT")
        submission_file = _get_submission_file_path(node_id, workspace)
        server_urls_str = os.environ["ML_MASTER_GRADING_SERVERS"]

        if not submission_file.exists():
            return json.dumps(
                {
                    "success": False,
                    "error": f"Submission file does not exist: {submission_file}",
                },
                ensure_ascii=False,
            )

        urls = [url.strip() for url in server_urls_str.split(",") if url.strip()]
        if not urls:
            return json.dumps(
                {
                    "success": False,
                    "error": "Grading server unavailable",
                    "details": "ML_MASTER_GRADING_SERVERS is empty",
                },
                ensure_ascii=False,
            )

        # 只使用唯一固定的 server url
        server_url = urls[0]

        # 第一步：先检查健康性
        if not _is_healthy(server_url):
            logger.warning(
                "Grading server unhealthy before validate, trying restart on same port: %s",
                server_url,
            )
            restarted_url = await asyncio.to_thread(
                ensure_server_fn, data_root, [server_url]
            )
            if restarted_url:
                server_url = restarted_url

        # 第二步：重启后再次检查健康性
        if not _is_healthy(server_url):
            return json.dumps(
                {
                    "success": False,
                    "error": "Grading server unavailable",
                    "details": "No healthy grading server found after restart",
                    "server_url": server_url,
                },
                ensure_ascii=False,
            )

        def _post_validate():
            with open(submission_file, "rb") as f:
                if _is_local_url(server_url):
                    with requests.Session() as session:
                        session.trust_env = False
                        return session.post(
                            f"{server_url}/validate",
                            files={"file": f},
                            headers={"exp-id": exp_id},
                            timeout=60,
                        )
                return requests.post(
                    f"{server_url}/validate",
                    files={"file": f},
                    headers={"exp-id": exp_id},
                    timeout=60,
                )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(_post_validate)

                if resp.status_code != 200:
                    logger.error(
                        "Grading server returned non-200 status (%s): %s",
                        resp.status_code,
                        resp.text,
                    )
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Grading server error",
                            "details": f"HTTP {resp.status_code}: {resp.text}",
                            "server_url": server_url,
                        },
                        ensure_ascii=False,
                    )

                try:
                    data = resp.json()
                except ValueError:
                    logger.error(
                        "Grading validate failed: response is not valid JSON. body=%r",
                        resp.text[:500],
                    )
                    return json.dumps(
                        {
                            "success": False,
                            "error": "Invalid JSON response from grading server",
                            "details": resp.text[:500],
                            "server_url": server_url,
                        },
                        ensure_ascii=False,
                    )

                if "error" in data:
                    logger.error("Grading server error: %s", data)
                    return json.dumps(
                        {
                            "success": False,
                            "error": data.get("error", "Grading server error"),
                            "details": data.get("details", ""),
                            "server_url": server_url,
                        },
                        ensure_ascii=False,
                    )

                return json.dumps(
                    {
                        "success": True,
                        "is_valid": data.get("is_valid", True),
                        "result": data.get("result"),
                        "details": data.get("details"),
                        "server_url": server_url,
                    },
                    ensure_ascii=False,
                )

            except requests.Timeout:
                logger.error(
                    "Grading validation timeout (%s), attempt %s/%s",
                    server_url,
                    attempt + 1,
                    max_retries,
                )
            except requests.RequestException as e:
                logger.error("Grading validation failed (%s): %s", server_url, e)

            # 重试前尝试同端口恢复一次
            restarted_url = await asyncio.to_thread(
                ensure_server_fn, data_root, [server_url]
            )
            if restarted_url:
                server_url = restarted_url

            await asyncio.sleep(1)

        return json.dumps(
            {
                "success": False,
                "error": "Grading server unavailable",
                "details": f"Validate failed after {max_retries} attempts",
                "server_url": server_url,
            },
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Validation failed: {str(e)}"},
            ensure_ascii=False,
        )

async def _grade_code_async(node_id: str, workspace: str, timeout: int, logger) -> str:
    """Shared implementation for grade_code tool."""
    try:
        data_root = os.environ["ML_MASTER_DATA_ROOT"]
        exp_id = os.environ["ML_MASTER_DATA_EXPID"]
        submission_file = _get_submission_file_path(node_id, workspace)

        if not submission_file.exists():
            return json.dumps(
                {
                    "success": False,
                    "error": f"Submission file does not exist: {submission_file}",
                },
                ensure_ascii=False,
            )

        # 优先使用 HTTP grade server（硬编码端口 7777）
        # 注意：这与 ML_MASTER_GRADING_SERVERS 不同，后者用于 validate_submission
        grade_server_url = "http://127.0.0.1:7777"

        # 先进行健康检查
        if await _check_grade_server_health(grade_server_url):
            logger.info(f"Using grade server at {grade_server_url}")
            return await _grade_via_http(grade_server_url, exp_id, str(submission_file), timeout, logger)
        else:
            logger.warning(f"Grade server at {grade_server_url} unavailable, falling back to subprocess")
            return await _grade_via_subprocess(data_root, exp_id, str(submission_file), timeout, logger)

    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"Grading failed: {str(e)}"}, ensure_ascii=False
        )


async def _grade_via_http(grade_server_url: str, exp_id: str, submission_path: str, timeout: int, logger) -> str:
    """通过 HTTP server 进行评分"""
    import aiohttp

    url = f"{grade_server_url.rstrip('/')}/grade"
    payload = {
        "exp_id": exp_id,
        "submission_path": submission_path,
        "timeout": timeout,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout + 10)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return json.dumps(
                        {"success": False, "error": f"Grade server returned status {resp.status}: {error_text}"},
                        ensure_ascii=False,
                    )

                data = await resp.json()

                # 转换响应格式以匹配原有的返回格式
                if data.get("success"):
                    output = (data.get("stdout") or "") + (data.get("stderr") or "")
                    return json.dumps(
                        {
                            "success": data.get("returncode") == 0,
                            "output": output,
                            "returncode": data.get("returncode"),
                        },
                        ensure_ascii=False,
                    )
                else:
                    return json.dumps(
                        {"success": False, "error": data.get("error", "Unknown error")},
                        ensure_ascii=False,
                    )

    except aiohttp.ClientError as e:
        logger.warning(f"Failed to connect to grade server at {grade_server_url}: {e}")
        return json.dumps(
            {"success": False, "error": f"Grade server unavailable: {str(e)}"},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"HTTP grading failed: {str(e)}"},
            ensure_ascii=False,
        )


async def _check_grade_server_health(grade_server_url: str, timeout: int = 3) -> bool:
    """检查 grade server 是否健康

    Args:
        grade_server_url: Grade server URL
        timeout: 超时时间（秒）

    Returns:
        True 如果健康，False 否则
    """
    import aiohttp

    health_url = f"{grade_server_url.rstrip('/')}/health"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("status") == "healthy"
                return False
    except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
        return False


async def _grade_via_subprocess(data_root: str, exp_id: str, submission_path: str, timeout: int, logger) -> str:
    """通过直接运行 subprocess 进行评分（回退方案）"""
    grade_script = Path(data_root) / exp_id / "prepared" / "grade.py"
    if not grade_script.exists():
        raise ValueError("grade.py does not exist!")

    private_dir = Path(data_root) / exp_id / "prepared" / "private"
    csv_files = sorted(private_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"Ground truth file does not exist in {private_dir}!")
    gt_file = csv_files[0]

    proc = await asyncio.create_subprocess_exec(
        "python",
        str(grade_script),
        "-g",
        str(gt_file),
        "-s",
        str(submission_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # 创建新进程组，便于 killpg 杀死整个进程树
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        # 杀死整个进程组
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass  # 进程已经结束
        await proc.wait()
        return json.dumps(
            {"success": False, "error": "Grading script execution timed out"},
            ensure_ascii=False,
        )

    output = stdout_bytes.decode("utf-8", errors="replace") + stderr_bytes.decode(
        "utf-8", errors="replace"
    )

    return json.dumps(
        {
            "success": proc.returncode == 0,
            "output": output,
            "returncode": proc.returncode,
        },
        ensure_ascii=False,
    )


# ========================================================================
# Sync wrapper functions for exp modules
# ========================================================================

def get_cached_execution_result(node_id: str, workspace: str) -> dict | None:
    """从 workspace/cache/{node_id}.json 获取缓存的执行结果。

    Args:
        node_id: 节点 ID
        workspace: 工作目录路径

    Returns:
        包含 stdout, exit_code, stderr, success, elapsed_time 的字典，如果不存在则返回 None
    """
    cache_file = Path(workspace) / "cache" / f"{node_id}.json"
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read cache file {cache_file}: {e}")
        return None


def run_code_sync(
    node_id: str,
    workspace: str,
    timeout: int = 300,
    load_resources_fn=None,  # 保留参数以向后兼容，但不再使用
) -> dict:
    """同步运行代码的包装函数。

    这是对 _run_code_async 的同步封装，用于 exp 模块中调用。
    执行结果会自动缓存到 workspace/cache/{node_id}.json。

    资源管理:
    - CPU: 自动选择占用最低的 32 个核心，使用 taskset 绑定
    - GPU: 自动选择 util 和显存占用最少的 GPU

    Args:
        node_id: 节点 ID
        workspace: 工作目录路径
        timeout: 超时时间（秒）
        load_resources_fn: 已废弃，保留仅为向后兼容

    Returns:
        包含 stdout, exit_code, stderr, success, elapsed_time, script, code 的字典
    """
    script_path = Path(workspace) / f"code_{node_id}.py"
    code = script_path.read_text(encoding="utf-8") if script_path.exists() else ""

    # Run async code in sync context (GPU selection and cache are handled inside _run_code_async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result_json = loop.run_until_complete(
            _run_code_async(node_id, workspace, timeout, None, logger)
        )
        result = json.loads(result_json)
    finally:
        loop.close()

    # Add script and code to result
    result["script"] = str(script_path)
    result["code"] = code

    return result


def grade_code_sync(node_id: str, workspace: str, timeout: int = 300, logger=None) -> dict:
    """同步版本的 grade 函数，供 Exp 模块使用。

    这是对 _grade_code_async 的同步封装，用于在 test-feedback 模式下获取测试集分数。

    Args:
        node_id: 节点 ID
        workspace: 工作目录路径
        timeout: 超时时间（秒），默认 300
        logger: 日志记录器（可选）

    Returns:
        包含 success, output, returncode 的字典
        - success (bool): 是否成功
        - output (str): grade 脚本的输出（stdout + stderr）
        - returncode (int): 退出码
        - error (str): 错误信息（仅在失败时）
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        # Run async grade in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_json = loop.run_until_complete(
                _grade_code_async(node_id, workspace, timeout, logger)
            )
            result = json.loads(result_json)
        finally:
            loop.close()

        if result.get("success"):
            output = result.get("output", "")
            return {
                "success": True,
                "output": output,
                "returncode": result.get("returncode")
            }
        return result
    except Exception as e:
        logger.warning(f"Grade sync failed for node {node_id}: {e}")
        return {"success": False, "error": str(e)}


def _validate_workspace(workspace: str) -> Path:
    """验证 workspace 是否存在，如果不存在则抛出 ValueError。

    Args:
        workspace: 工作目录路径（字符串）

    Returns:
        Path: 转换后的 Path 对象

    Raises:
        ValueError: 当 workspace 不存在时
    """
    workspace_path = Path(workspace).resolve()
    if not workspace_path.exists():
        raise ValueError(
            f"Workspace 不存在：{workspace_path}\n"
            f"请提供正确的 **WorkSpace 绝对路径**。\n"
        )
    return workspace_path