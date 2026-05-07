#!/usr/bin/env python3
"""
批量配置 MLE-Bench 实验脚本

从 result.csv 读取 75 个比赛列表，批量生成配置文件并更新 run_mle_maintable.sh
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path


# 固定的 initial instruction
INITIAL_INSTRUCTION = (
    "Attention! You are allowed to finetune the hyperparameters of the given initial node, "
    "the core mission for you is to select a scalable algorithm that can gain better performance "
    "when getting more augmented data. (But for current initial state, you can only use the raw data)"
)

# 默认路径
PROJECT_ROOT = Path(__file__).parent.parent
RESULT_CSV = PROJECT_ROOT / "result.csv"
BATCH_YAML = PROJECT_ROOT / "scripts" / "main_table_batch.yaml"
INITIAL_CODE_DIR = PROJECT_ROOT / "initial_code" / "algoonly"
TASK_BASE_DIR = Path("${DATA_ROOT}")
CONFIG_DIR = PROJECT_ROOT / "configs" / "ml_master_datatree"
RUN_SCRIPT = PROJECT_ROOT / "run_mle_maintable.sh"


def load_batch_config(yaml_path):
    """从 YAML 文件加载批量配置"""
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["main_table"]


def load_competitions(csv_path):
    """从 CSV 文件加载比赛列表"""
    competitions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            competitions.append({
                "name": row["competition"],
                "direction": row["metric_direction"],  # higher or lower
                "category": row["category"],
            })
    return competitions


def load_mcp_templates(config_dir):
    """从基础 JSON 文件加载 MCP 配置模板"""
    templates = {}

    # 加载 base_initial.json
    initial_path = config_dir / "base_initial.json"
    if initial_path.exists():
        with open(initial_path, "r", encoding="utf-8") as f:
            templates["initial"] = json.load(f)
    else:
        raise FileNotFoundError(f"找不到基础配置文件: {initial_path}")

    # 加载 base_all.json
    all_path = config_dir / "base_all.json"
    if all_path.exists():
        with open(all_path, "r", encoding="utf-8") as f:
            templates["node_all"] = json.load(f)
    else:
        raise FileNotFoundError(f"找不到基础配置文件: {all_path}")

    # 加载 base_datanode.json
    datanode_path = config_dir / "base_datanode.json"
    if datanode_path.exists():
        with open(datanode_path, "r", encoding="utf-8") as f:
            templates["data_node"] = json.load(f)
    else:
        raise FileNotFoundError(f"找不到基础配置文件: {datanode_path}")

    return templates


def load_yaml_template(config_dir):
    """从 config_base.yaml 加载 YAML 配置模板"""
    yaml_path = config_dir / "config_base.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"找不到基础配置文件: {yaml_path}")


def set_mcp_env(mcp_config, data_root, exp_id, grading_server):
    """设置 MCP 配置中的环境变量"""
    for server_config in mcp_config["mcpServers"].values():
        if "env" in server_config:
            env = server_config["env"]
            if "ML_MASTER_DATA_ROOT" in env:
                env["ML_MASTER_DATA_ROOT"] = data_root
            if "ML_MASTER_DATA_EXPID" in env:
                env["ML_MASTER_DATA_EXPID"] = exp_id
            if "ML_MASTER_GRADING_SERVERS" in env:
                env["ML_MASTER_GRADING_SERVERS"] = grading_server
    return mcp_config


def get_grading_server(exp_name, port_start=50000):
    """为每个实验分配一个端口（使用简单的哈希）"""
    # 使用实验名称的哈希值来分配端口，避免冲突
    # 使用 50000 以上高位端口
    hash_val = abs(hash(exp_name)) % 10000
    return f"http://127.0.0.1:{port_start + hash_val}"


def generate_mcp_configs(config_dir, exp_name, data_root, grading_server):
    """生成三个 MCP 配置文件"""
    # 加载 MCP 模板
    mcp_templates = load_mcp_templates(config_dir)

    # 创建 json_configs 子目录
    json_configs_dir = config_dir / "json_configs" / exp_name
    json_configs_dir.mkdir(parents=True, exist_ok=True)

    # 生成 initial MCP 配置
    initial_config = set_mcp_env(
        copy.deepcopy(mcp_templates["initial"]), data_root, exp_name, grading_server
    )
    initial_filename = f"initial_{exp_name}.json"
    initial_path = json_configs_dir / initial_filename
    with open(initial_path, "w", encoding="utf-8") as f:
        json.dump(initial_config, f, indent=2, ensure_ascii=False)

    # 生成 node_all MCP 配置
    node_all_config = set_mcp_env(
        copy.deepcopy(mcp_templates["node_all"]), data_root, exp_name, grading_server
    )
    node_all_filename = f"node_all_{exp_name}.json"
    node_all_path = json_configs_dir / node_all_filename
    with open(node_all_path, "w", encoding="utf-8") as f:
        json.dump(node_all_config, f, indent=2, ensure_ascii=False)

    # 生成 data_node MCP 配置
    data_node_config = set_mcp_env(
        copy.deepcopy(mcp_templates["data_node"]), data_root, exp_name, grading_server
    )
    data_node_filename = f"data_node_{exp_name}.json"
    data_node_path = json_configs_dir / data_node_filename
    with open(data_node_path, "w", encoding="utf-8") as f:
        json.dump(data_node_config, f, indent=2, ensure_ascii=False)

    return {
        "initial": f"../../json_configs/{exp_name}/{initial_filename}",
        "node_all": f"../../json_configs/{exp_name}/{node_all_filename}",
        "data_node": f"../../json_configs/{exp_name}/{data_node_filename}",
    }


def generate_yaml_config(
    config_dir, exp_name, mcp_files, data_root, grading_server, llm_config, max_steps
):
    """生成 YAML 配置文件"""
    from string import Template

    # 加载 YAML 模板
    yaml_template = load_yaml_template(config_dir)

    # 使用 Template 替换模板中的占位符
    template = Template(yaml_template)
    yaml_content = template.substitute(
        exp_id=exp_name,
        data_root=data_root,
        grading_server=grading_server,
        mcp_initial=mcp_files["initial"],
        mcp_node_all=mcp_files["node_all"],
        mcp_data_node=mcp_files["data_node"],
        llm_model=llm_config["model"],
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
        max_steps=max_steps,
    )

    yaml_configs_dir = config_dir / "yaml_configs" / exp_name
    yaml_configs_dir.mkdir(parents=True, exist_ok=True)

    config_filename = f"config_{exp_name}.yaml"
    config_path = yaml_configs_dir / config_filename

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # 返回相对于 configs/ml_master_datatree 的路径
    return f"yaml_configs/{exp_name}/{config_filename}"


def update_run_mle_maintable_sh(
    script_path,
    competitions_config,
):
    """批量更新 run_mle_maintable.sh 脚本"""
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 生成所有 case 条目
    case_entries = []
    for config in competitions_config:
        exp_name = config["exp_name"]
        config_name = config["config_name"]
        task_path = config["task_path"]
        initial_code = config["initial_code"]
        force_direction = config["force_direction"]

        case_lines = [
        f'    "{exp_name}")',
        f"        python run.py \\",
        f"            --agent ml_master_datatree \\",
        f"            --config configs/ml_master_datatree/{config_name} \\",
        f"            --task {task_path} \\",
        f"            --initial-code {initial_code} \\",
        f'            --initial-instruction "{INITIAL_INSTRUCTION}" \\',
        f"            --test-feedback \\",
        f"            {force_direction}",
        f"        ;;",
        ""
        ]
        case_entries.append("\n".join(case_lines))

    all_cases = "\n".join(case_entries)

    # 使用占位符替换
    new_content = content.replace(
        "# AUTO_GENERATED_CASES_PLACEHOLDER",
        all_cases.strip()
    )

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    parser = argparse.ArgumentParser(description="批量配置 MLE-Bench 实验")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要生成的配置，不实际写入文件",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的比赛数量（用于测试）",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="过滤比赛名称（只处理包含该字符串的比赛）",
    )
    args = parser.parse_args()

    # 加载配置
    print("加载配置文件...")
    batch_config = load_batch_config(BATCH_YAML)
    competitions = load_competitions(RESULT_CSV)

    llm_config = {
        "model": batch_config["model"],
        "api_key": batch_config["api_key"],
        "base_url": batch_config["base_url"],
    }
    max_steps = batch_config["max_steps"]
    data_root = str(TASK_BASE_DIR)

    print(f"  - LLM 模型: {llm_config['model']}")
    print(f"  - Max Steps: {max_steps}")
    print(f"  - 数据根目录: {data_root}")
    print(f"  - 比赛数量: {len(competitions)}")

    # 应用过滤
    if args.filter:
        competitions = [c for c in competitions if args.filter in c["name"]]
        print(f"  - 过滤后数量: {len(competitions)}")

    if args.limit:
        competitions = competitions[:args.limit]
        print(f"  - 限制处理后数量: {len(competitions)}")

    # 收集所有配置信息
    all_configs = []
    missing_task_files = []
    fatal_errors = []

    for comp in competitions:
        exp_name = comp["name"]
        direction = comp["direction"]

        # 路径配置
        initial_code_path = INITIAL_CODE_DIR / f"algoonly_{exp_name}.py"
        task_path = TASK_BASE_DIR / exp_name / "prepared" / "public" / "description.md"

        # 检查文件是否存在
        # 初始代码文件必须存在，否则终止
        if not initial_code_path.exists():
            fatal_errors.append(f"初始代码文件不存在: {initial_code_path}")
        # 任务描述文件可以不存在，只警告
        if not task_path.exists():
            missing_task_files.append(f"任务描述: {task_path}")

        # force direction
        force_direction = "--force-maximize" if direction == "higher" else "--force-minimize"

        # grading server
        grading_server = get_grading_server(exp_name)

        config_info = {
            "exp_name": exp_name,
            "config_name": f"yaml_configs/{exp_name}/config_{exp_name}.yaml",
            "task_path": str(task_path),
            "initial_code": str(initial_code_path),
            "force_direction": force_direction,
            "data_root": data_root,
            "grading_server": grading_server,
            "llm_config": llm_config,
            "max_steps": max_steps,
        }
        all_configs.append(config_info)

    # 如果有致命错误，终止程序
    if fatal_errors:
        print("\n✗ 错误: 缺少必要文件，程序终止:")
        for f in fatal_errors:
            print(f"  - {f}")
        return 1

    # 如果任务描述文件缺失，只输出警告
    if missing_task_files:
        print("\n⚠ 警告: 以下任务描述文件不存在（将忽略）:")
        for f in missing_task_files[:10]:
            print(f"  - {f}")
        if len(missing_task_files) > 10:
            print(f"  ... 还有 {len(missing_task_files) - 10} 个文件")

    # Dry run 模式
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - 将要生成的配置:")
        print("=" * 60)
        for config in all_configs[:5]:  # 只显示前 5 个
            print(f"\n实验: {config['exp_name']}")
            print(f"  Config: {config['config_name']}")
            print(f"  Task: {config['task_path']}")
            print(f"  Initial Code: {config['initial_code']}")
            print(f"  Force: {config['force_direction']}")
        if len(all_configs) > 5:
            print(f"\n... 还有 {len(all_configs) - 5} 个配置")
        return 0

    # 确认开始
    print(f"\n{'='*60}")
    print(f"将为 {len(all_configs)} 个比赛生成配置")
    print(f"{'='*60}")
    response = input("是否开始配置？(y/n): ").strip().lower()
    if response != "y":
        print("已取消")
        return 0

    # 检查配置目录是否存在
    if not CONFIG_DIR.exists():
        print(f"错误: 配置目录不存在: {CONFIG_DIR}")
        return 1

    # 生成配置文件
    print("\n生成配置文件...")
    success_count = 0
    for i, config in enumerate(all_configs):
        exp_name = config["exp_name"]
        try:
            # 1. 生成 MCP 配置
            mcp_files = generate_mcp_configs(
                CONFIG_DIR,
                exp_name,
                config["data_root"],
                config["grading_server"],
            )

            # 2. 生成 YAML 配置
            config_name = generate_yaml_config(
                CONFIG_DIR,
                exp_name,
                mcp_files,
                config["data_root"],
                config["grading_server"],
                config["llm_config"],
                config["max_steps"],
            )

            # 更新 config_name 为相对路径
            config["config_name"] = config_name

            print(f"  [{i+1}/{len(all_configs)}] ✓ {exp_name}")
            success_count += 1
        except Exception as e:
            print(f"  [{i+1}/{len(all_configs)}] ✗ {exp_name}: {e}")

    # 更新 run_mle_maintable.sh
    print(f"\n更新 {RUN_SCRIPT.name}...")
    try:
        update_run_mle_maintable_sh(RUN_SCRIPT, all_configs)
        print(f"  ✓ 已更新 {RUN_SCRIPT}")
    except Exception as e:
        print(f"  ✗ 更新失败: {e}")
        return 1

    print(f"\n{'='*60}")
    print(f"✓ 配置完成! 成功: {success_count}/{len(all_configs)}")
    print(f"{'='*60}")
    print(f"\n运行实验:")
    print(f"bash run_mle_maintable.sh <experiment_name>")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())