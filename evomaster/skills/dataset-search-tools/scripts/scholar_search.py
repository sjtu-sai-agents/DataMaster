#!/usr/bin/env python3
"""Scholar search CLI - wraps search_scholar.py

Usage:
    python scholar_search.py arxiv          --query "benchmark dataset" -n 10
    python scholar_search.py arxiv-author   --author "Yann LeCun" -n 5
    python scholar_search.py scholar        --query "sentiment dataset"
    python scholar_search.py dblp-papers    --query "Diffusion Model" -n 5
    python scholar_search.py dblp-authors   --query "Geoffrey Hinton" -n 5
    python scholar_search.py dblp-venues    --query "CVPR" -n 5
"""
import argparse
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_scholar import (
    arxiv_search_by_content,
    arxiv_search_by_author,
    google_scholar_search,
    search_dblp_papers,
    search_dblp_authors,
    search_dblp_venues,
)


def main():
    parser = argparse.ArgumentParser(description="Scholar search CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("arxiv", help="Search arXiv by content")
    p.add_argument("--query", "-q", required=True, help="Search query")
    p.add_argument("-n", type=int, default=5, help="Max results (default: 5)")

    p = sub.add_parser("arxiv-author", help="Search arXiv by author")
    p.add_argument("--author", "-a", required=True, help="Author name")
    p.add_argument("-n", type=int, default=5, help="Max results (default: 5)")

    p = sub.add_parser("scholar", help="Google Scholar search")
    p.add_argument("--query", "-q", required=True, help="Search query")

    p = sub.add_parser("dblp-papers", help="Search DBLP papers")
    p.add_argument("--query", "-q", required=True, help="Search query")
    p.add_argument("-n", type=int, default=5, help="Max results (default: 5)")

    p = sub.add_parser("dblp-authors", help="Search DBLP authors")
    p.add_argument("--query", "-q", required=True, help="Search query")
    p.add_argument("-n", type=int, default=5, help="Max results (default: 5)")

    p = sub.add_parser("dblp-venues", help="Search DBLP venues")
    p.add_argument("--query", "-q", required=True, help="Search query")
    p.add_argument("-n", type=int, default=5, help="Max results (default: 5)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "arxiv":
        result = arxiv_search_by_content(query_string=args.query, max_results=args.n)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "arxiv-author":
        result = arxiv_search_by_author(author_name=args.author, max_results=args.n)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "scholar":
        print(google_scholar_search(query=args.query))

    elif args.command == "dblp-papers":
        print(asyncio.run(search_dblp_papers(query=args.query, max_results=args.n)))

    elif args.command == "dblp-authors":
        print(asyncio.run(search_dblp_authors(query=args.query, max_results=args.n)))

    elif args.command == "dblp-venues":
        print(asyncio.run(search_dblp_venues(query=args.query, max_results=args.n)))


if __name__ == "__main__":
    main()
