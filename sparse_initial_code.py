import json
from pathlib import Path

if __name__ == "__main__":
    file_path = "playground/ml_master_datatree/initial_code.json"
    with open(file_path, "r", encoding="utf-8") as file:
        contents: dict = json.load(file)

    initial_code_path = Path("playground/ml_master_datatree/initial_code")

    for competition_id, code in contents.items():
        if not code:
            print(f"Error: {competition_id} code is empty")
            continue
        
        current_path = initial_code_path / Path(competition_id)
        current_path.mkdir(exist_ok=True, parents=True)
        with open(
            (current_path / Path("initial_code.py")), "w", encoding="utf-8"
        ) as file:
            file.write(code)
