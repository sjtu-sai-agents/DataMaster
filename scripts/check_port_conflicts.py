#!/usr/bin/env python3
"""
检查 yaml 配置文件中的 grading_servers 端口冲突
"""

import os
import re
from collections import defaultdict
from pathlib import Path
import yaml


def extract_port_from_url(url: str) -> int | None:
    """从 URL 中提取端口号"""
    match = re.search(r':(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def find_all_yaml_files(yaml_dir: str) -> list[Path]:
    """递归查找所有 yaml 文件"""
    yaml_path = Path(yaml_dir)
    if not yaml_path.exists():
        raise FileNotFoundError(f"目录不存在: {yaml_dir}")

    return list(yaml_path.rglob("*.yaml"))


def check_port_conflicts(yaml_dir: str) -> dict:
    """
    检查所有 yaml 文件中的 grading_servers 端口冲突

    返回:
        dict: {
            "port_usage": {port: [(file_path, url), ...]},
            "conflicts": {port: [(file_path, url), ...]},
            "no_grading_servers": [file_path, ...],
            "total_files": int,
            "files_with_grading_servers": int
        }
    """
    yaml_files = find_all_yaml_files(yaml_dir)

    port_usage = defaultdict(list)  # port -> [(file, url), ...]
    no_grading_servers = []

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)

            if not isinstance(content, dict):
                continue

            grading_servers = content.get('grading_servers')

            if grading_servers is None:
                no_grading_servers.append(str(yaml_file.relative_to(yaml_dir)))
                continue

            if not isinstance(grading_servers, list):
                print(f"警告: {yaml_file.relative_to(yaml_dir)} 中的 grading_servers 不是列表类型")
                continue

            for server_url in grading_servers:
                port = extract_port_from_url(server_url)
                if port:
                    port_usage[port].append((str(yaml_file.relative_to(yaml_dir)), server_url))

        except Exception as e:
            print(f"解析文件失败 {yaml_file.relative_to(yaml_dir)}: {e}")

    # 找出冲突的端口
    conflicts = {port: files for port, files in port_usage.items() if len(files) > 1}

    return {
        "port_usage": dict(port_usage),
        "conflicts": conflicts,
        "no_grading_servers": no_grading_servers,
        "total_files": len(yaml_files),
        "files_with_grading_servers": len(yaml_files) - len(no_grading_servers)
    }


def print_results(results: dict, verbose: bool = True):
    """打印检查结果"""
    print("=" * 80)
    print("端口冲突检查结果")
    print("=" * 80)

    print(f"\n总文件数: {results['total_files']}")
    print(f"包含 grading_servers 的文件数: {results['files_with_grading_servers']}")
    print(f"不包含 grading_servers 的文件数: {len(results['no_grading_servers'])}")

    if results['conflicts']:
        print(f"\n{'='*80}")
        print(f"发现 {len(results['conflicts'])} 个端口冲突!")
        print("=" * 80)
        for port, files in sorted(results['conflicts'].items()):
            print(f"\n端口 {port} 被以下 {len(files)} 个文件使用:")
            for file_path, url in files:
                print(f"  - {file_path}")
                print(f"    URL: {url}")
    else:
        print(f"\n{'='*80}")
        print("未发现端口冲突!")
        print("=" * 80)

    if verbose and results['port_usage']:
        print(f"\n{'='*80}")
        print("所有端口使用情况:")
        print("=" * 80)
        for port in sorted(results['port_usage'].keys()):
            files = results['port_usage'][port]
            print(f"端口 {port}: {len(files)} 个文件")

    if results['no_grading_servers']:
        print(f"\n{'='*80}")
        print(f"未配置 grading_servers 的文件 ({len(results['no_grading_servers'])} 个):")
        print("=" * 80)
        for file_path in sorted(results['no_grading_servers']):
            print(f"  - {file_path}")

    print("\n" + "=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="检查 yaml 配置文件中的 grading_servers 端口冲突")
    parser.add_argument(
        "yaml_dir",
        nargs="?",
        default="configs/ml_master_datatree/yaml_configs",
        help="yaml 配置文件目录路径"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式，只显示冲突信息")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    parser.add_argument(
        "--base-dir",
        default="${PROJECT_ROOT}",
        help="项目根目录"
    )

    args = parser.parse_args()

    # 构建完整路径
    yaml_dir = os.path.join(args.base_dir, args.yaml_dir)

    if not os.path.exists(yaml_dir):
        print(f"错误: 目录不存在: {yaml_dir}")
        return 1

    results = check_port_conflicts(yaml_dir)

    if args.json:
        import json
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print_results(results, verbose=not args.quiet)

    # 返回码: 0 表示无冲突, 1 表示有冲突
    return 1 if results['conflicts'] else 0


if __name__ == "__main__":
    exit(main())
