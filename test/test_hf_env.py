#!/usr/bin/env python3
"""测试 HuggingFace 环境变量是否正确设置"""

import os
from huggingface_hub import whoami

print("=" * 60)
print("检查 HuggingFace 环境变量")
print("=" * 60)

# 检查环境变量
env_vars = {
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "HF_ENDPOINT": os.getenv("HF_ENDPOINT"),
    "HF_HOME": os.getenv("HF_HOME"),
    "HF_DATASETS_CACHE": os.getenv("HF_DATASETS_CACHE"),
}

for var_name, var_value in env_vars.items():
    status = "✅ 已设置" if var_value else "❌ 未设置"
    print(f"{var_name}: {status}")
    if var_value:
        # 隐藏 token 的敏感部分
        if "TOKEN" in var_name:
            masked = var_value[:10] + "..." if len(var_value) > 10 else "***"
            print(f"  值: {masked}")
        else:
            print(f"  值: {var_value}")

print("\n" + "=" * 60)
print("测试 HuggingFace 认证")
print("=" * 60)

# 测试认证
try:
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        user_info = whoami()
        print(f"✅ 认证成功！用户: {user_info}")
    else:
        print("❌ HF_TOKEN 未设置，无法测试认证")
except Exception as e:
    print(f"❌ 认证失败: {e}")

print("\n" + "=" * 60)
print("当前 HuggingFace 配置")
print("=" * 60)

# 打印实际使用的缓存目录
from datasets import config as datasets_config
print(f"Datasets 缓存目录: {datasets_config.HF_DATASETS_CACHE}")
print(f"HUB 缓存目录: {datasets_config.HF_HUB_CACHE}")
