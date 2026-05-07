#!/bin/bash

PORT=8899
REMOTE_PORT=8890

if lsof -i :$PORT > /dev/null; then
    echo "有进程占用本地 $PORT 端口"
    lsof -i :$PORT
else
    echo "没有进程占用本地 $PORT 端口"
    echo "启动 Auto SSH 服务 (Local:$PORT -> Remote:$REMOTE_PORT)"
    # 修改这里：将远程的 REMOTE_PORT 映射到本地的 $PORT
    autossh -M 0 -N -L $PORT:localhost:$REMOTE_PORT Voc_node1
fi