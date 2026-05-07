# Grade HTTP 服务器使用指南

## 概述

`grade_code` 工具现在已经支持通过 HTTP 服务器来执行评分操作，避免每次都创建新的 Python subprocess，从而降低 CPU 负载。

## 架构

```
Agent 调用 grade_code
    ↓
_grade_code_async 检查环境变量 ML_MASTER_GRADE_SERVER
    ↓
    ├─ 如果设置了 → 发送 HTTP 请求到 grade server
    └─ 如果未设置 → 回退到直接运行 subprocess（原有方式）
```

## 启动 Grade HTTP 服务器

### 方式 1：直接启动

```bash
python initialize_grade_port.py \
    --data-root ${DATA_ROOT} \
    --host 127.0.0.1 \
    --port 5004 \
    --workers 1
```

### 方式 2：后台启动

```bash
nohup python initialize_grade_port.py \
    --data-root ${DATA_ROOT} \
    --port 5004 \
    > grade_server.log 2>&1 &
```

### 方式 3：使用 systemd（推荐）

创建 `/etc/systemd/system/ml-master-grade.service`:

```ini
[Unit]
Description=ML Master Grade HTTP Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=${PROJECT_ROOT}
Environment="ML_MASTER_DATA_ROOT=${DATA_ROOT}"
ExecStart=/usr/bin/python ${PROJECT_ROOT}/initialize_grade_port.py --port 5004
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl start ml-master-grade
sudo systemctl enable ml-master-grade
```

## 配置 Agent 使用 Grade Server

在启动 Agent 或 Playground 时，设置环境变量：

```bash
export ML_MASTER_GRADE_SERVER="http://127.0.0.1:5004"
```

或在 Python 中：
```python
import os
os.environ["ML_MASTER_GRADE_SERVER"] = "http://127.0.0.1:5004"
```

## API 文档

### POST /grade

对提交文件进行评分。

**请求:**
```json
{
    "exp_id": "leaf-classification",
    "submission_path": "/path/to/submission.csv",
    "timeout": 300
}
```

**响应:**
```json
{
    "success": true,
    "stdout": "metric = 0.95678\n...",
    "stderr": "",
    "returncode": 0
}
```

### GET /health

健康检查。

**响应:**
```json
{
    "status": "healthy",
    "service": "grade-server"
}
```

## 测试

### 测试服务器是否正常

```bash
curl http://127.0.0.1:5004/health
```

### 测试评分功能

```bash
curl -X POST http://127.0.0.1:5004/grade \
    -H "Content-Type: application/json" \
    -d '{
        "exp_id": "leaf-classification",
        "submission_path": "/path/to/submission.csv",
        "timeout": 300
    }'
```

## 回退机制

如果没有设置 `ML_MASTER_GRADE_SERVER` 环境变量，或者 grade server 不可用，系统会自动回退到原有的 subprocess 方式，保证兼容性。

## 性能优化

- **线程池**: 服务器使用 4 个线程池处理并发请求
- **Worker 进程**: 可以使用 `--workers` 参数启动多进程
- **连接复用**: 使用 aiohttp 的连接池

## 监控

查看日志：
```bash
tail -f grade_server.log
```

检查服务状态：
```bash
sudo systemctl status ml-master-grade
```

## 故障排查

### 问题：Agent 调用 grade_code 超时

检查 grade server 是否运行：
```bash
curl http://127.0.0.1:5004/health
```

### 问题：连接被拒绝

确认端口没有被占用：
```bash
lsof -i :5004
```

确认环境变量设置正确：
```bash
echo $ML_MASTER_GRADE_SERVER
```

### 问题：评分返回错误

检查 grade.py 文件是否存在：
```bash
ls ${DATA_ROOT}/<exp_id>/prepared/grade.py
```

检查 ground truth 文件是否存在：
```bash
ls ${DATA_ROOT}/<exp_id>/prepared/private/*.csv
```