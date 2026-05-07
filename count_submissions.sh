#!/bin/bash
# Find the run folder with the most files in workspaces/task_0/submission/

RUNS_DIR="${PROJECT_ROOT}/runs"
max_count=0
max_folder=""

for folder in "$RUNS_DIR"/*/; do
    submission_dir="${folder}workspaces/task_0/submission"
    if [ -d "$submission_dir" ]; then
        count=$(ls -1 "$submission_dir" 2>/dev/null | wc -l)
        if [ "$count" -gt "$max_count" ]; then
            max_count=$count
            max_folder=$(basename "$folder")
        fi
    fi
done

echo "Folder with most submissions: $max_folder"
echo "Number of files: $max_count"
