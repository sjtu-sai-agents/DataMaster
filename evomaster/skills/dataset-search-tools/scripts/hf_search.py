#!/usr/bin/env python3
"""HuggingFace dataset search CLI - wraps search_huggingface.py

Usage:
    python hf_search.py search --query "sentiment" --limit 10
    python hf_search.py search --query "translation" --author Helsinki-NLP
    python hf_search.py inspect --id stanfordnlp/sst2
    python hf_search.py inspect --id stanfordnlp/sst2 --config default
    python hf_search.py configs --id glue
    python hf_search.py splits --id stanfordnlp/sst2
    python hf_search.py sample  --id stanfordnlp/sst2 -n 5 --split train
    python hf_search.py readme  --id stanfordnlp/sst2
    python hf_search.py download --id stanfordnlp/sst2 -o ./data_links/sst2
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_huggingface import (
    search_datasets,
    inspect_dataset,
    get_dataset_configs,
    get_dataset_splits,
    get_dataset_sample,
    get_dataset_readme,
    download_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="HuggingFace dataset search CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("search", help="Search datasets by keyword")
    p.add_argument("--query", "-q", required=True, help="Search keyword")
    p.add_argument("--limit", "-l", type=int, default=100, help="Max results (default: 100)")
    p.add_argument("--author", "-a", default=None, help="Filter by author/org")

    p = sub.add_parser("inspect", help="Inspect dataset metadata")
    p.add_argument("--id", required=True, help="Dataset ID (e.g. stanfordnlp/sst2)")
    p.add_argument("--config", "-c", default=None, help="Config name")

    p = sub.add_parser("configs", help="List available configs")
    p.add_argument("--id", required=True, help="Dataset ID")

    p = sub.add_parser("splits", help="List available splits")
    p.add_argument("--id", required=True, help="Dataset ID")
    p.add_argument("--config", "-c", default=None, help="Config name")

    p = sub.add_parser("sample", help="Preview sample records")
    p.add_argument("--id", required=True, help="Dataset ID")
    p.add_argument("--config", "-c", default=None, help="Config name")
    p.add_argument("--split", "-s", default=None, help="Split name")
    p.add_argument("-n", type=int, default=5, help="Number of samples (default: 5)")

    p = sub.add_parser("readme", help="Get dataset README")
    p.add_argument("--id", required=True, help="Dataset ID")

    p = sub.add_parser("download", help="Download dataset files")
    p.add_argument("--id", required=True, help="Dataset ID")
    p.add_argument("--output", "-o", required=True, help="Output directory")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "search":
        print(search_datasets(query=args.query, limit=args.limit, author=args.author))
    elif args.command == "inspect":
        print(inspect_dataset(dataset_id=args.id, config=args.config))
    elif args.command == "configs":
        result = get_dataset_configs(dataset_id=args.id)
        print("\n".join(result) if isinstance(result, list) else result)
    elif args.command == "splits":
        print(get_dataset_splits(dataset_id=args.id, config=args.config))
    elif args.command == "sample":
        print(get_dataset_sample(dataset_id=args.id, config=args.config, split=args.split, num_samples=args.n))
    elif args.command == "readme":
        print(get_dataset_readme(dataset_id=args.id))
    elif args.command == "download":
        print(download_dataset(dataset_id=args.id, output_dir=args.output))


if __name__ == "__main__":
    main()
