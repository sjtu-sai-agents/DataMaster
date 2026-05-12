#!/usr/bin/env python3
"""
自动配置 ML Master DataTree 实验的配置文件
用法: python scripts/auto_config_exp.py <exp_name> <task_path> [--port PORT] [--data_root ROOT]
"""

import json
import os
import re
import sys
from pathlib import Path


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


def generate_mcp_configs(config_dir, exp_name, exp_suffix, data_root, grading_server):
    """生成三个 MCP 配置文件"""
    import copy

    # 加载 MCP 模板
    mcp_templates = load_mcp_templates(config_dir)

    # 创建 json_configs 子目录（使用基础实验名称，不带后缀）
    # 如果有后缀，去掉后缀部分；否则使用完整名称
    if exp_suffix and exp_name.endswith(f"_{exp_suffix}"):
        base_exp_name = exp_name[:-(len(exp_suffix) + 1)]  # 去掉 "_{suffix}"
    else:
        base_exp_name = exp_name

    json_configs_dir = config_dir / "json_configs" / base_exp_name
    json_configs_dir.mkdir(parents=True, exist_ok=True)

    # 生成 initial MCP 配置
    # 注意：ML_MASTER_DATA_EXPID 使用 base_exp_name（去掉后缀的纯净竞赛名称）
    initial_config = set_mcp_env(
        copy.deepcopy(mcp_templates["initial"]), data_root, base_exp_name, grading_server
    )
    # 文件名使用完整的 exp_name（可能包含后缀）来区分不同配置
    initial_filename = f"initial_{exp_name}.json"
    initial_path = json_configs_dir / initial_filename
    with open(initial_path, "w", encoding="utf-8") as f:
        json.dump(initial_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 创建: {initial_path}")

    # 生成 node_all MCP 配置
    node_all_config = set_mcp_env(
        copy.deepcopy(mcp_templates["node_all"]), data_root, base_exp_name, grading_server
    )
    node_all_filename = f"node_all_{exp_name}.json"
    node_all_path = json_configs_dir / node_all_filename
    with open(node_all_path, "w", encoding="utf-8") as f:
        json.dump(node_all_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 创建: {node_all_path}")

    # 生成 data_node MCP 配置
    data_node_config = set_mcp_env(
        copy.deepcopy(mcp_templates["data_node"]), data_root, base_exp_name, grading_server
    )
    data_node_filename = f"data_node_{exp_name}.json"
    data_node_path = json_configs_dir / data_node_filename
    with open(data_node_path, "w", encoding="utf-8") as f:
        json.dump(data_node_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 创建: {data_node_path}")

    return {
        "initial": f"../../json_configs/{base_exp_name}/{initial_filename}",
        "node_all": f"../../json_configs/{base_exp_name}/{node_all_filename}",
        "data_node": f"../../json_configs/{base_exp_name}/{data_node_filename}",
    }


def generate_yaml_config(
    config_dir, exp_name, exp_suffix, mcp_files, data_root, grading_server, llm_config, max_steps
):
    """生成 YAML 配置文件"""
    from string import Template

    # 加载 YAML 模板
    yaml_template = load_yaml_template(config_dir)
    
    # 创建 yaml_configs 子目录（使用基础实验名称，不带后缀）
    # 如果有后缀，去掉后缀部分；否则使用完整名称
    if exp_suffix and exp_name.endswith(f"_{exp_suffix}"):
        base_exp_name = exp_name[:-(len(exp_suffix) + 1)]  # 去掉 "_{suffix}"
    else:
        base_exp_name = exp_name

    # 使用 Template 替换模板中的占位符
    template = Template(yaml_template)
    yaml_content = template.substitute(
        exp_id=base_exp_name,
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

    yaml_configs_dir = config_dir / "yaml_configs" / base_exp_name
    yaml_configs_dir.mkdir(parents=True, exist_ok=True)

    # 使用更简洁的配置文件名（文件名使用完整的 exp_name 来区分不同配置）
    config_filename = f"config_{exp_name}.yaml"
    config_path = yaml_configs_dir / config_filename

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"✓ 创建: {config_path}")

    # 返回相对于 configs/ml_master_datatree 的路径
    return f"yaml_configs/{base_exp_name}/{config_filename}"


def update_run_mle_sh(
    script_path,
    exp_name,
    config_name,
    task_path,
    initial_code=None,
    initial_instruction=None,
):
    """更新 run_mle.sh 脚本"""
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 生成新的 case 条目
    new_case_lines = [
        f'    "{exp_name}")',
        f"        python run.py \\",
        f"            --agent ml_master_datatree \\",
        f"            --config configs/ml_master_datatree/{config_name} \\",
        f"            --task {task_path} \\",
    ]

    # 添加可选参数
    if initial_code:
        new_case_lines.append(f"            --initial-code {initial_code} \\")
    if initial_instruction:
        new_case_lines.append(
            f"            --initial-instruction {initial_instruction} \\"
        )

    # 闭合命令
    new_case_lines.append(f"        ;;")

    # 检查是否已经存在该实验
    if f'"{exp_name}")' in content:
        print(f"⚠ run_mle.sh 中已存在实验 '{exp_name}'，跳过更新")
        return False

    # 在 *) 之前插入新的 case
    lines = content.split("\n")
    insert_index = None
    for i, line in enumerate(lines):
        if line.strip() == "*)":
            insert_index = i
            break

    if insert_index is not None:
        # 在 *) 之前插入（确保有空行）
        lines.insert(insert_index, "")
        # 插入新 case 的内容（反向插入，保持顺序）
        for case_line in reversed(new_case_lines):
            lines.insert(insert_index, case_line)

        new_content = "\n".join(lines)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✓ 更新: {script_path}")
        return True
    else:
        print("⚠ 无法找到 *) 标记，跳过更新 run_mle.sh")
        return False


def check_port_available(port):
    """检查端口是否可用"""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("127.0.0.1", port))
    sock.close()
    return result != 0


def get_available_port(start_port=5003):
    """获取可用端口"""
    port = start_port
    while port < 65535:
        if check_port_available(port):
            return port
        port += 1
    raise ValueError("无法找到可用端口")


def interactive_input():
    """交互式获取用户输入"""
    print("\n" + "=" * 60)
    print("ML Master DataTree 实验配置工具")
    print("=" * 60 + "\n")

    # 实验名称
    while True:
        exp_name = input("请输入实验名称 (如: leaf-classification): ").strip()
        if exp_name:
            if not re.match(r"^[a-zA-Z0-9_-]+$", exp_name):
                print("⚠ 实验名称只能包含字母、数字、下划线和连字符")
                continue
            break
        print("⚠ 实验名称不能为空")

    # 任务路径
    while True:
        task_path = input(
            "请输入任务描述文件路径 (如: /data/.../description.md): "
        ).strip()
        if task_path:
            if not Path(task_path).exists():
                print(f"⚠ 警告: 文件不存在: {task_path}")
                response = input("是否继续？(y/n): ").strip().lower()
                if response == "y":
                    break
            else:
                break
        else:
            print("⚠ 任务路径不能为空")

    # 端口
    default_port = 5003
    while True:
        port_input = input(
            f"请输入 Grading 服务器端口 (默认: {default_port}): "
        ).strip()
        if not port_input:
            port = default_port
        else:
            try:
                port = int(port_input)
                if port < 1 or port > 65535:
                    print("⚠ 端口必须在 1-65535 之间")
                    continue
            except ValueError:
                print("⚠ 请输入有效的端口号")
                continue

        # 检查端口是否可用
        if check_port_available(port):
            print(f"✓ 端口 {port} 可用")
            break
        else:
            print(f"⚠ 端口 {port} 已被占用")
            response = input("是否自动查找可用端口？(y/n): ").strip().lower()
            if response == "y":
                try:
                    port = get_available_port(port)
                    print(f"✓ 自动分配端口: {port}")
                    break
                except ValueError as e:
                    print(f"⚠ {e}")
                    continue
            else:
                continue

    # 数据根目录
    default_data_root = "${DATA_ROOT}"
    data_root_input = input(f"请输入数据根目录 (默认: {default_data_root}): ").strip()
    data_root = data_root_input if data_root_input else default_data_root

    # 可选参数
    initial_code = None
    initial_instruction = None
    exp_suffix = None

    print("\n可选参数:")
    add_optional = input("是否添加初始代码或指令？(y/n): ").strip().lower()
    if add_optional == "y":
        # 初始代码
        code_path = input("请输入初始代码文件路径 (留空跳过): ").strip()
        if code_path and Path(code_path).exists():
            initial_code = code_path
            print(f"✓ 将使用初始代码: {code_path}")
        elif code_path:
            print(f"⚠ 文件不存在，跳过初始代码")

        # 初始指令
        instruction_path = input("请输入初始指令文件路径 (留空跳过): ").strip()
        if instruction_path and Path(instruction_path).exists():
            initial_instruction = instruction_path
            print(f"✓ 将使用初始指令: {instruction_path}")
        elif instruction_path:
            print(f"⚠ 文件不存在，跳过初始指令")

        # 如果添加了初始代码或指令，询问是否添加后缀
        if initial_code or initial_instruction:
            print("\n实验名称后缀配置:")
            add_suffix = input("是否为实验名称添加后缀以区分不同的配置？(y/n): ").strip().lower()
            if add_suffix == "y":
                while True:
                    suffix = input("请输入后缀 (如: v2, test, custom): ").strip()
                    if suffix:
                        if not re.match(r"^[a-zA-Z0-9_-]+$", suffix):
                            print("⚠ 后缀只能包含字母、数字、下划线和连字符")
                            continue
                        exp_suffix = suffix
                        print(f"✓ 实验名称将添加后缀: {suffix}")
                        break
                    print("⚠ 后缀不能为空")

    # LLM 配置
    default_llm_model = "Vendor2/GPT-5.3-codex"
    default_llm_api_key = "${LLM_API_KEY}"
    default_llm_base_url = "${LLM_BASE_URL}"
    
    print("\nLLM 配置:")
    print(f"默认 LLM 配置: {default_llm_model}\n{default_llm_api_key}\n{default_llm_base_url}")
    use_default_llm = input(f"是否使用默认 LLM 配置？(y/n): ").strip().lower()


    if use_default_llm == "y":
        llm_model = default_llm_model
        llm_api_key = default_llm_api_key
        llm_base_url = default_llm_base_url
        print(f"✓ 使用默认 LLM 配置")
    else:
        # 自定义 LLM 配置
        llm_model_input = input(
            f"请输入 LLM 模型 (默认: {default_llm_model}): "
        ).strip()
        llm_model = llm_model_input if llm_model_input else default_llm_model

        llm_api_key_input = input(
            f"请输入 API Key (默认: {default_llm_api_key}): "
        ).strip()
        llm_api_key = llm_api_key_input if llm_api_key_input else default_llm_api_key

        llm_base_url_input = input(
            f"请输入 Base URL (默认: {default_llm_base_url}): "
        ).strip()
        llm_base_url = (
            llm_base_url_input if llm_base_url_input else default_llm_base_url
        )

        print(f"✓ 使用自定义 LLM 配置")

    # max_steps 配置
    default_max_steps = 150
    max_steps_input = input(f"请输入 max_steps (默认: {default_max_steps}): ").strip()
    max_steps = int(max_steps_input) if max_steps_input else default_max_steps
    print(f"✓ max_steps 设置为: {max_steps}")

    return {
        "exp_name": exp_name,
        "task_path": task_path,
        "port": port,
        "data_root": data_root,
        "initial_code": initial_code,
        "initial_instruction": initial_instruction,
        "exp_suffix": exp_suffix,
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "max_steps": max_steps,
    }


def main():
    # 获取交互式输入
    params = interactive_input()
    exp_name = params["exp_name"]
    task_path = params["task_path"]
    port = params["port"]
    data_root = params["data_root"]
    initial_code = params["initial_code"]
    initial_instruction = params["initial_instruction"]
    exp_suffix = params["exp_suffix"]
    llm_config = {
        "model": params["llm_model"],
        "api_key": params["llm_api_key"],
        "base_url": params["llm_base_url"],
    }
    max_steps = params["max_steps"]

    # 如果有后缀，则添加到实验名称
    if exp_suffix:
        exp_name_with_suffix = f"{exp_name}_{exp_suffix}"
    else:
        exp_name_with_suffix = exp_name

    # 设置路径
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs" / "ml_master_datatree"
    run_mle_sh = project_root / "run_mle.sh"

    grading_server = f"http://127.0.0.1:{port}"

    print(f"\n{'='*60}")
    print(f"配置实验: {exp_name_with_suffix}")
    print(f"{'='*60}")
    print(f"实验名称: {exp_name_with_suffix}")
    if exp_suffix:
        print(f"  (基础名称: {exp_name}, 后缀: {exp_suffix})")
    print(f"任务路径: {task_path}")
    print(f"数据根目录: {data_root}")
    print(f"评分服务器: {grading_server}")
    print(f"LLM 模型: {llm_config['model']}")
    print(f"LLM Base URL: {llm_config['base_url']}")
    print(f"Max Steps: {max_steps}")
    if initial_code:
        print(f"初始代码: {initial_code}")
    if initial_instruction:
        print(f"初始指令: {initial_instruction}")
    print(f"配置目录: {config_dir}")
    print(f"{'='*60}\n")

    # 确认开始
    response = input("是否开始配置？(y/n): ").strip().lower()
    if response != "y":
        print("已取消")
        return 0

    # 检查配置目录是否存在
    if not config_dir.exists():
        print(f"错误: 配置目录不存在: {config_dir}")
        return 1

    try:
        # 1. 生成 MCP 配置文件
        print("步骤 1/3: 生成 MCP 配置文件")
        mcp_files = generate_mcp_configs(
            config_dir, exp_name_with_suffix, exp_suffix, data_root, grading_server
        )
        print()

        # 2. 生成 YAML 配置文件
        print("步骤 2/3: 生成 YAML 配置文件")
        config_name = generate_yaml_config(
            config_dir,
            exp_name_with_suffix,
            exp_suffix,
            mcp_files,
            data_root,
            grading_server,
            llm_config,
            max_steps,
        )
        print()

        # 3. 更新 run_mle.sh
        print("步骤 3/3: 更新 run_mle.sh")
        update_run_mle_sh(
            run_mle_sh,
            exp_name_with_suffix,
            config_name,
            task_path,
            initial_code,
            initial_instruction,
        )
        print()

        print(f"{'='*60}")
        print("✓ 配置完成！")
        print(f"{'='*60}")
        print(f"\n运行实验:")
        print(f"bash run_mle.sh {exp_name_with_suffix}")
        print()

        return 0

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
