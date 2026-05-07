import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict
import re

# Add agentcodebase to path
sys.path.insert(0, "${PROJECT_ROOT}/agentcodebase")
from codebase import LLMPipeline


BASE_DIR = "${PROJECT_ROOT}/initial_codebase"
CONFIG_PATH = "${PROJECT_ROOT}/agentcodebase/config/config.yaml"


def extract_code_content(file_path: str) -> str:
    """Read and return the content of a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def parse_llm_response(response: str) -> Dict[str, str]:
    """Parse the LLM response to extract algo_abstract and algo_detail."""
    algo_abstract = ""
    algo_detail = ""

    # Extract algo_abstract
    abstract_match = re.search(r'<algo_abstract>\s*(.*?)\s*</algo_abstract>', response, re.DOTALL)
    if abstract_match:
        algo_abstract = abstract_match.group(1).strip()

    # Extract algo_detail
    detail_match = re.search(r'<algo_detail>\s*(.*?)\s*</algo_detail>', response, re.DOTALL)
    if detail_match:
        algo_detail = detail_match.group(1).strip()

    # Extract file_id (hash) if present
    file_id = ""
    id_match = re.search(r'<file_id>\s*(.*?)\s*</file_id>', response, re.DOTALL)
    if id_match:
        file_id = id_match.group(1).strip()

    return {
        "algo_abstract": algo_abstract,
        "algo_detail": algo_detail,
        "file_id": file_id
    }


def generate_file_id(file_name: str, code_preview: str) -> str:
    """Generate a unique ID for a file based on name and content preview."""
    content = f"{file_name}:{code_preview[:200]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def test_single_file():
    """Test with just ONE file to verify the fix works."""
    import yaml

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Override prompts for algorithm analysis
    config['llm_pipeline']['prompts']['system_prompt_path'] = "${PROJECT_ROOT}/agentcodebase/prompts/system/algorithm_analysis.md"
    config['llm_pipeline']['prompts']['user_prompt_path'] = "${PROJECT_ROOT}/agentcodebase/prompts/user/algorithm_analysis.md"

    # Save temporary config
    temp_config_path = "/tmp/algorithm_analysis_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    pipeline = LLMPipeline(config_path=temp_config_path)

    # Get first solution file from first directory
    exp_dir = Path(BASE_DIR) / "jigsaw-toxic-comment-classification-challenge"
    solution_files = list(exp_dir.glob("solution_*.py"))

    if not solution_files:
        print("No solution files found!")
        return

    # Test with just 2 files
    test_files = solution_files[:2]

    print(f"Testing with {len(test_files)} files...")

    files_data = []
    for sol_file in test_files:
        code_content = extract_code_content(str(sol_file))
        if not code_content:
            continue

        if len(code_content) > 10000:
            code_content = code_content[:10000] + "\n# ... (truncated)"

        file_id = generate_file_id(sol_file.name, code_content)

        files_data.append({
            "code": code_content,
            "file_name": sol_file.name,
            "file_id": file_id
        })

    # Process files
    file_id_map = {fd["file_id"]: fd for fd in files_data}
    data_pool = [{
        "user_prompt_kwargs": {"code": fd["code"], "file_id": fd["file_id"]},
        "system_prompt_kwargs": {}
    } for fd in files_data]

    results_list = pipeline.run(
        data_pool,
        concurrency_limit=2,
        extract_function=lambda x: x
    )

    print(f"\nGot {len(results_list)} results from pipeline")

    for i, result_dict in enumerate(results_list):
        print(f"\n--- Result {i+1} ---")
        print(f"Keys: {result_dict.keys()}")

        if "error" in result_dict:
            print(f"ERROR: {result_dict['error']}")
            continue

        # Get the response text
        response_text = result_dict.get("response", "")
        print(f"Response type: {type(response_text)}")

        if isinstance(response_text, dict):
            print(f"Response dict keys: {response_text.keys()}")
            response_text = response_text.get("content", str(response_text))

        print(f"Response preview (first 200 chars): {str(response_text)[:200]}...")

        parsed = parse_llm_response(response_text)
        print(f"Parsed file_id: {parsed.get('file_id', 'N/A')}")
        print(f"Algo abstract: {parsed.get('algo_abstract', 'N/A')[:100]}...")

    # Clean up
    os.remove(temp_config_path)


if __name__ == "__main__":
    test_single_file()
