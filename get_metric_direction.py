#!/usr/bin/env python3
"""
Script to get metric direction from result.csv based on exp_id from config.yaml.
"""
import sys
import os
import yaml


def get_exp_id_from_yaml(run_dir):
    """Extract exp_id from config.yaml in the run directory."""
    config_path = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}", file=sys.stderr)
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            exp_id = config.get('exp_id')
            if exp_id:
                return exp_id
            else:
                print(f"Error: exp_id not found in {config_path}", file=sys.stderr)
                return None
    except Exception as e:
        print(f"Error reading config.yaml: {e}", file=sys.stderr)
        return None


def get_metric_direction(exp_id, result_csv_path):
    """Get metric direction from result.csv for given exp_id."""
    if not os.path.exists(result_csv_path):
        print(f"Error: result.csv not found at {result_csv_path}", file=sys.stderr)
        return None

    try:
        with open(result_csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(f'{exp_id},'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        return parts[1].strip().lower()
    except Exception as e:
        print(f"Error reading result.csv: {e}", file=sys.stderr)
        return None

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: get_metric_direction.py <RUN_DIR> [RESULT_CSV_PATH]", file=sys.stderr)
        print("  RUN_DIR: Path to the run directory", file=sys.stderr)
        print("  RESULT_CSV_PATH: Optional path to result.csv", file=sys.stderr)
        sys.exit(1)

    run_dir = sys.argv[1]
    result_csv_path = sys.argv[2] if len(sys.argv) >= 3 else "${PROJECT_ROOT}/result.csv"

    # Get exp_id from config.yaml
    exp_id = get_exp_id_from_yaml(run_dir)
    if not exp_id:
        sys.exit(1)

    print(f"Info: Found exp_id: {exp_id}", file=sys.stderr)

    # Get metric direction from result.csv
    metric_direction = get_metric_direction(exp_id, result_csv_path)
    if not metric_direction:
        print(f"Error: Could not find metric direction for exp_id '{exp_id}' in {result_csv_path}", file=sys.stderr)
        sys.exit(1)

    # Validate
    if metric_direction not in ['lower', 'higher']:
        print(f"Error: Invalid metric direction '{metric_direction}' for exp_id '{exp_id}'", file=sys.stderr)
        sys.exit(1)

    # Output only the direction (can be captured by shell)
    print(f"Info: Metric direction: {metric_direction}", file=sys.stderr)
    print(metric_direction)


if __name__ == "__main__":
    main()
