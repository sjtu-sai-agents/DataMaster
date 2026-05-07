#!/bin/bash
# 缓存配置脚本 - 用于 math_posttrain_datatree 实验

# ============================================================
# HuggingFace 数据集缓存配置
# ============================================================

# HF_HOME: HuggingFace 主缓存目录 (已配置)
export HF_HOME=/data/HF_Cache_dataevo

# HF_ENDPOINT: 使用镜像加速下载
export HF_ENDPOINT=https://hf-mirror.com

# HF_DATASETS_CACHE: datasets 库的缓存位置
# 默认: $HF_HOME/datasets
# 当前已使用: 374GB
export HF_DATASETS_CACHE=/data/HF_Cache_dataevo/datasets

echo "✓ HuggingFace Cache Configuration:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_ENDPOINT: $HF_ENDPOINT"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"

# ============================================================
# Math PostTrain 数据树特定缓存
# ============================================================

# MATH_PT_SHARED_CACHE: 共享的已物化数据集缓存
# 设置后，所有实验都会复用这个缓存，避免重复下载和处理
export MATH_PT_SHARED_CACHE=/data/yaxindu/datascientist/math_posttrain_cache

echo ""
echo "✓ Math PostTrain Cache Configuration:"
echo "  MATH_PT_SHARED_CACHE: $MATH_PT_SHARED_CACHE"

# 创建共享缓存目录
mkdir -p "$MATH_PT_SHARED_CACHE/materialized_datasets"
echo "  Created: $MATH_PT_SHARED_CACHE/materialized_datasets"

# ============================================================
# 可选配置
# ============================================================

# 强制使用 datasets-server API (不推荐，除非调试)
# export MATH_PT_FORCE_DATASETS_SERVER=1

# 完全禁用 datasets-server 回退 (如果镜像稳定可以启用)
# export MATH_PT_DISABLE_DATASETS_SERVER=1

echo ""
echo "✓ Cache setup complete!"
echo ""
echo "Usage:"
echo "  source playground/math_posttrain_datatree/setup_cache.sh"
echo ""
echo "Check cache usage:"
echo "  du -sh $HF_DATASETS_CACHE"
echo "  du -sh $MATH_PT_SHARED_CACHE"
