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


def analyze_files_batch(files_data: List[Dict[str, str]], pipeline: LLMPipeline, concurrency_limit: int = 10) -> List[Dict[str, str]]:
    """Analyze multiple files in parallel using the LLM pipeline."""
    if not files_data:
        return []

    print(f"  Analyzing {len(files_data)} files in parallel (concurrency: {concurrency_limit})...")

    # Create a mapping from file_id to file data for correct matching
    file_id_map = {}
    data_pool = []

    for file_data in files_data:
        file_id = file_data["file_id"]
        file_id_map[file_id] = file_data
        data_pool.append({
            "user_prompt_kwargs": {
                "code": file_data["code"],
                "file_id": file_id
            },
            "system_prompt_kwargs": {}
        })

    try:
        results_list = pipeline.run(
            data_pool,
            concurrency_limit=concurrency_limit,
            extract_function=lambda x: x
        )

        # Match responses using file_id
        results = []
        matched_ids = set()

        for result_dict in results_list:
            # Check if there's an error
            if "error" in result_dict:
                print(f"  Error in pipeline result: {result_dict['error']}")
                continue

            # Get the response text from the result dict (it's already a string)
            response_text = result_dict.get("response", "")

            parsed = parse_llm_response(response_text)
            response_id = parsed.get("file_id", "")

            if response_id and response_id in file_id_map:
                # Match found by file_id
                file_data = file_id_map[response_id]
                results.append({
                    "code": file_data["code"],
                    "algo_abstract": parsed["algo_abstract"],
                    "algo_detail": parsed["algo_detail"],
                    "file_name": file_data["file_name"]
                })
                matched_ids.add(response_id)
            else:
                # No match found - this shouldn't happen with proper LLM responses
                print(f"  Warning: Could not match response with file_id: {response_id}")

        # Check for any unmatched files
        unmatched = set(file_id_map.keys()) - matched_ids
        if unmatched:
            print(f"  Warning: {len(unmatched)} files did not get matched responses")

        return results

    except Exception as e:
        print(f"  Error during batch analysis: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return error entries
        results = []
        for file_data in files_data:
            results.append({
                "code": file_data["code"],
                "algo_abstract": f"Error: {str(e)}",
                "algo_detail": f"Batch processing failed, error: {str(e)}",
                "file_name": file_data["file_name"]
            })
        return results


def process_experiment_directory(exp_dir: str, pipeline: LLMPipeline, max_files: int = None, concurrency_limit: int = 10) -> List[Dict[str, str]]:
    """Process all solution files in an experiment directory using parallel processing."""
    solution_files = list(Path(exp_dir).glob("solution_*.py"))

    if max_files:
        solution_files = solution_files[:max_files]

    print(f"Found {len(solution_files)} solution files in {exp_dir}")

    # Load all files first
    files_data = []
    for sol_file in solution_files:
        code_content = extract_code_content(str(sol_file))
        if not code_content:
            continue

        # Truncate code if too long
        if len(code_content) > 10000:
            code_content = code_content[:10000] + "\n# ... (truncated)"

        # Generate unique file_id
        file_id = generate_file_id(sol_file.name, code_content)

        files_data.append({
            "code": code_content,
            "file_name": sol_file.name,
            "file_id": file_id
        })

    # Process all files in parallel
    results = analyze_files_batch(files_data, pipeline, concurrency_limit=concurrency_limit)

    return results


def main():
    """Main function to process all experiment directories."""
    # Initialize LLM pipeline with custom prompts
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

    # Get all experiment directories
    exp_dirs = []
    for item in Path(BASE_DIR).iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains solution files
            solution_files = list(item.glob("solution_*.py"))
            if solution_files:
                exp_dirs.append(item)

    print(f"Found {len(exp_dirs)} experiment directories to process")

    # Process each experiment directory
    for exp_dir in exp_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {exp_dir.name}")
        print(f"{'='*60}")

        # Process all solution files in parallel (concurrency: 10)
        results = process_experiment_directory(str(exp_dir), pipeline, max_files=None, concurrency_limit=10)

        # Generate info.json
        info_json_path = exp_dir / "info.json"

        info_data = []
        for result in results:
            info_data.append({
                "code": result["code"],
                "algo_abstract": result["algo_abstract"],
                "algo_detail": result["algo_detail"]
            })

        # Save info.json
        with open(info_json_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Generated {info_json_path} with {len(info_data)} entries")

    # Clean up temp config
    os.remove(temp_config_path)

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
