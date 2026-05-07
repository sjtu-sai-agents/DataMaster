from pathlib import Path
import re

def save_code_to_file(directory, filename, code_content):
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = target_dir / filename

    file_path.write_text(code_content, encoding='utf-8')
    
    print(f"文件已成功保存至: {file_path}")


def replace_submission_name (code, _id):
    submission_file_name = f"submission_{_id}.csv"
    modified_code = code
    if "submission/submission.csv" in code:
        modified_code = code.replace("submission/submission.csv", f"submission/{submission_file_name}")
    if "/submission.csv" in modified_code:
        modified_code = modified_code.replace("/submission.csv", f"/{submission_file_name}")

    if "to_csv('submission.csv" in modified_code:
        modified_code = modified_code.replace("to_csv('submission.csv", f"to_csv('submission/{submission_file_name}")
    if 'to_csv("submission.csv' in modified_code:
        modified_code = modified_code.replace('to_csv("submission.csv', f'to_csv("submission/{submission_file_name}')

    if '"submission.csv"' in modified_code:
        modified_code = modified_code.replace('"submission.csv"', f'"{submission_file_name}"')
    if "'submission.csv'" in modified_code:
        modified_code = modified_code.replace("'submission.csv'", f"'{submission_file_name}'")
    
    return modified_code

def read_code(value: str, _id: str) -> str:
    """Extract code if value contains a markdown code block; otherwise return original."""
    match = re.search(r"```(?:python)?\s*(.*?)\s*```", value, re.DOTALL)
    if match:
        value = match.group(1).strip()
    return replace_submission_name(value, _id), value




