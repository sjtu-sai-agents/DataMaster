# HF Sandbox Server

HuggingFace API 请求代理服务，部署在 CPU 机器上，为 GPU 训练机提供统一的限速、排队、缓存能力，避免高并发场景下触发 429。

## 架构

```
GPU 机器                              CPU 机器
┌──────────────────────┐              ┌──────────────────────┐
│ MCP 工具 / data.py   │── SSH 隧道 ──>│ hf_sandbox_server    │
│                      │  (port 8899) │   ├─ 令牌桶限速 30RPM │
│ HF_SANDBOX_URL=      │              │   ├─ 并发控制 max=3   │
│  http://localhost:8899│              │   ├─ 搜索缓存 10min   │
└──────────────────────┘              │   └─ hf-mirror.com   │
                                      └──────────────────────┘
```

Sandbox 是**可选的**：`HF_SANDBOX_URL` 有值时走代理，为空或不设时走原有 hf-mirror 直连，零影响。

## 文件说明

| 文件 | 说明 |
|------|------|
| `server.py` | FastAPI 主服务，定义所有端点、限速、缓存逻辑 |
| `hf_ops.py` | 实际 HF 操作（搜索/inspect/下载/物化），从原有代码提取 |
| `rate_limiter.py` | 令牌桶限速器 |
| `config.py` | 配置管理（所有参数通过环境变量） |
| `start_sandbox.sh` | 服务启停脚本（start/stop/restart/status） |
| `requirements.txt` | Python 依赖 |

## API 端点

| 端点 | 方法 | 说明 | 超时 |
|------|------|------|------|
| `/health` | GET | 健康检查 + 运行指标 | 5s |
| `/search` | POST | 搜索数据集（有缓存） | 30s |
| `/inspect` | POST | 数据集元信息 | 30s |
| `/configs` | POST | 获取 configs | 30s |
| `/splits` | POST | 获取 splits | 30s |
| `/readme` | POST | 获取 README | 30s |
| `/sample` | POST | 获取样本行 | 120s |
| `/download` | POST | 下载到 NFS | 600s |
| `/materialize` | POST | 物化数据集到 NFS | 600s |

## 部署

### 1. 在 CPU 机器上启动服务

```bash
cd ${WORKSPACE_ROOT}/DataScientistEvomaster2 && ./hf_sandbox_server/start_sandbox.sh start
```

其他命令：
```bash
./hf_sandbox_server/start_sandbox.sh stop      # 停止
./hf_sandbox_server/start_sandbox.sh restart   # 重启
./hf_sandbox_server/start_sandbox.sh status    # 查看状态
```

### 2. 在 GPU 机器上建立 SSH 隧道

CPU 机器的 8899 端口不对外暴露，需要通过 SSH 隧道转发：

```bash
# 前提：已配置 SSH 别名 cpu3（见下方 SSH 配置）
ssh -fN -L 8899:localhost:8899 cpu3
```

SSH 配置（`/root/.ssh/config`）：
```
Host cpu3
    HostName 115.190.3.255
    Port 32072
    User root
    IdentityFile ~/.ssh/cpu3_ed25519
    StrictHostKeyChecking no
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### 3. 验证

```bash
# 健康检查
curl http://localhost:8899/health

# 测试搜索
curl -X POST http://localhost:8899/search \
  -H "Content-Type: application/json" \
  -d '{"query": "math", "limit": 3}'
```

## 配置

所有参数通过环境变量控制：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF API 端点 |
| `HF_TOKEN` | `""` | HuggingFace token |
| `HF_RATE_LIMIT_RPM` | `30` | 每分钟最大请求数 |
| `HF_MAX_CONCURRENT` | `3` | 最大并发 HF 请求 |
| `HF_SEARCH_CACHE_TTL` | `600` | 搜索缓存 TTL（秒） |
| `SANDBOX_HOST` | `0.0.0.0` | 监听地址 |
| `SANDBOX_PORT` | `8899` | 监听端口 |

## 客户端配置（GPU 机器侧）

在以下两个文件中设置 `HF_SANDBOX_URL`：

**`configs/math_posttrain_datatree/config.yaml`**：
```yaml
data_access:
  hf_sandbox_url: "http://localhost:8899"  # 留空则走直连
```

**`configs/math_posttrain_datatree/mcp_config4data_isolated.json`**：
```json
"HF_SANDBOX_URL": "http://localhost:8899"
```

### 禁用 Sandbox

把上述两处的值改为空字符串 `""`，所有请求回退到 hf-mirror 直连，不需要改代码。
