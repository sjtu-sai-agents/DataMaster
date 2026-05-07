#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试 HuggingFace 搜索工具的代理连接
"""

import sys
import os

# 添加搜索工具路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'search_dataset_tools'))

# 导入搜索工具
from search_huggingface import search_datasets

def test_search():
    """测试搜索功能"""
    print("=" * 60)
    print("测试 HuggingFace 搜索工具")
    print("=" * 60)

    # 显示当前代理设置
    print("\n当前代理设置:")
    print(f"  http_proxy: {os.environ.get('http_proxy', '未设置')}")
    print(f"  https_proxy: {os.environ.get('https_proxy', '未设置')}")
    print(f"  all_proxy: {os.environ.get('all_proxy', '未设置')}")
    print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")

    print("\n开始测试搜索...")
    try:
        # 测试一个简单的搜索
        result = search_datasets(
            query="AIME competition math",
            limit=10
        )

        print("\n✅ 搜索成功！")
        print("\n搜索结果:")
        print("-" * 60)
        print(result)
        print("-" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 搜索失败！")
        print(f"错误信息: {str(e)}")
        print(f"错误类型: {type(e).__name__}")

        import traceback
        print("\n完整错误堆栈:")
        traceback.print_exc()

        return False

if __name__ == "__main__":
    success = test_search()
    sys.exit(0 if success else 1)
