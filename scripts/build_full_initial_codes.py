#!/usr/bin/env python3
"""
为 ablation 实验准备：拼接 data_loader.py 和 algo.py 生成 full_code.py
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path("${PROJECT_ROOT}")
DATA_LOADER_FORMAT_DIR = PROJECT_ROOT / "initial_code" / "data_loader_format"


def build_full_code_for_dataset(dataset_path: Path) -> bool:
    """
    为单个 dataset 生成 full_code.py

    Args:
        dataset_path: dataset 目录路径

    Returns:
        bool: 是否成功生成
    """
    # 检查必需文件
    algo_file = dataset_path / "algo.py"
    data_loader_file = dataset_path / "data_loader.py"
    output_file = dataset_path / "full_code.py"

    if not algo_file.exists():
        print(f"  ❌ 缺少 algo.py")
        return False

    if not data_loader_file.exists():
        print(f"  ❌ 缺少 data_loader.py")
        return False

    # 读取文件内容
    try:
        with open(data_loader_file, 'r', encoding='utf-8') as f:
            data_loader_content = f.read()

        with open(algo_file, 'r', encoding='utf-8') as f:
            algo_content = f.read()

    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")
        return False

    # 拼接：data_loader.py + 两个换行 + algo.py
    full_code_content = data_loader_content + "\n\n" + algo_content

    # 写入 full_code.py
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_code_content)
        return True
    except Exception as e:
        print(f"  ❌ 写入文件失败: {e}")
        return False


def main():
    """主函数：遍历所有 dataset 目录并生成 full_code.py"""
    print("=" * 60)
    print("开始为所有 dataset 生成 full_code.py")
    print("=" * 60)
    print()

    if not DATA_LOADER_FORMAT_DIR.exists():
        print(f"❌ 目录不存在: {DATA_LOADER_FORMAT_DIR}")
        return

    # 获取所有子目录
    dataset_dirs = [d for d in DATA_LOADER_FORMAT_DIR.iterdir() if d.is_dir()]

    if not dataset_dirs:
        print(f"❌ 没有找到 dataset 目录")
        return

    print(f"找到 {len(dataset_dirs)} 个 dataset 目录")
    print()

    # 统计
    success_count = 0
    failed_count = 0
    failed_datasets = []

    # 遍历所有 dataset
    for dataset_dir in sorted(dataset_dirs):
        print(f"处理 {dataset_dir.name}...")

        if build_full_code_for_dataset(dataset_dir):
            print(f"  ✅ 成功生成 full_code.py")
            success_count += 1
        else:
            print(f"  ❌ 生成失败")
            failed_count += 1
            failed_datasets.append(dataset_dir.name)

        print()

    # 输出总结
    print("=" * 60)
    print("生成完成")
    print("=" * 60)
    print(f"✅ 成功: {success_count}/{len(dataset_dirs)}")
    print(f"❌ 失败: {failed_count}/{len(dataset_dirs)}")

    if failed_datasets:
        print(f"\n失败的 dataset:")
        for name in failed_datasets:
            print(f"  - {name}")


if __name__ == "__main__":
    main()