import os
from playground.ml_master.core.utils.grading import validate_submission

sub_dir = "/data/yaxindu/datascientist/DataScientistEvomaster2/runs/ml_master_20260309_034732/workspaces/task_0/submission"

for file in os.listdir(sub_dir):
    if file.endswith(".csv"):
        submission_path = os.path.join(sub_dir, file)
        ok, result = validate_submission(
            exp_id="detecting-insults-in-social-commentary",
            submission_path=submission_path,
            server_urls=["http://127.0.0.1:5003"],
            dataset_root="/data/public_data/exp_data/demo1bench",
            timeout=60,
            max_retries=3,
        )
        print(f"Validation result for {file}: ok={ok}, result={result}")