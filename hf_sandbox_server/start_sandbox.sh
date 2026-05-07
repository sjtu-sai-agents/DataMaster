#!/usr/bin/env bash
set -euo pipefail

# ============================================
# HF Sandbox Server 服务管理脚本
# ============================================
#
# 用法:
#   ./start_sandbox.sh              # 启动服务
#   ./start_sandbox.sh stop         # 停止服务
#   ./start_sandbox.sh restart      # 重启服务
#   ./start_sandbox.sh status       # 查看状态
#
# 环境变量(可选):
#   SANDBOX_PORT        服务端口 (默认 8899)
#   SANDBOX_HOST        监听地址 (默认 0.0.0.0)
#   HF_ENDPOINT         HF 镜像地址 (默认 https://hf-mirror.com)
#   HF_TOKEN            HuggingFace token
#   HF_RATE_LIMIT_RPM   每分钟最大请求数 (默认 30)
#   HF_MAX_CONCURRENT   最大并发请求数 (默认 3)
#   HF_SEARCH_CACHE_TTL 搜索缓存 TTL 秒 (默认 600)
#   PYTHON              Python 解释器路径

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${SCRIPT_DIR}/.pids"
LOG_DIR="${SCRIPT_DIR}/.logs"

PORT="${SANDBOX_PORT:-8899}"
HOST="${SANDBOX_HOST:-0.0.0.0}"
PYTHON="${PYTHON:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

mkdir -p "${PID_DIR}" "${LOG_DIR}"

start() {
    local pid_file="${PID_DIR}/sandbox.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            log_info "sandbox 已在运行 (PID: ${pid}, 端口: ${PORT})"
            return 0
        fi
    fi

    log_info "启动 HF Sandbox Server (${HOST}:${PORT}) ..."
    cd "${SCRIPT_DIR}/.."

    # 保存 PID
    echo $$ > "${pid_file}"

    # 前台运行，实时输出到终端并保存到日志
    SANDBOX_PORT="${PORT}" SANDBOX_HOST="${HOST}" \
    "${PYTHON}" -m hf_sandbox_server.server \
        2>&1 | tee "${LOG_DIR}/sandbox.log"
}

stop() {
    local pid_file="${PID_DIR}/sandbox.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            log_info "已停止 sandbox (PID: ${pid})"
        else
            log_warn "sandbox 进程 ${pid} 不存在"
        fi
        rm -f "${pid_file}"
    else
        log_warn "sandbox PID 文件不存在"
    fi
}

status() {
    echo "============================================"
    echo "HF Sandbox Server 状态"
    echo "============================================"
    local pid_file="${PID_DIR}/sandbox.pid"
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid=$(cat "${pid_file}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo -e "  ${GREEN}●${NC} sandbox: 运行中 (PID: ${pid}, 端口: ${PORT})"
        else
            echo -e "  ${RED}○${NC} sandbox: 未运行 (stale PID: ${pid})"
        fi
    else
        echo -e "  ${RED}○${NC} sandbox: 未运行"
    fi
    echo "============================================"
    echo ""
    echo "端点:  http://${HOST}:${PORT}/health"
    echo "日志:  ${LOG_DIR}/sandbox.log"
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 2; start ;;
    status)  status ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
