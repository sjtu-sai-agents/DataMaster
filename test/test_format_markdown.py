config_format = {
    "node_id": 100,
    "workspace": "/data/12345",
    "task_description": "123",
    "best_code": "123",
    "previous_code": "123",
    "execution_output": "123",
    "parent_dataloader": "123",
    "memory": "123",
    "data_loader_readme": "123",
    "parent_dataloader": "123",
    "data_preview": "123",
    "operation_tools_readme": "123",
    "memory_tree_manual": "123",
    "best_metric": "123",
}

format_base_file = "playground/ml_master_datatree/prompts/black/user_prompt.md"

with open(format_base_file, "r", encoding="utf-8") as file:
    content = file.read()
    
content = content.format(**config_format)

print(content)
