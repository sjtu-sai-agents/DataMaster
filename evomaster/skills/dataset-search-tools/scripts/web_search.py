#!/usr/bin/env python3
"""Web search CLI - wraps search_web.py

Usage:
    python web_search.py search --query "public NLP dataset download"
    python web_search.py parse  --url "https://example.com/page"
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search_web import google_search, web_parse


def main():
    parser = argparse.ArgumentParser(description="Web search CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("search", help="Google search")
    p.add_argument("--query", "-q", required=True, help="Search query")

    p = sub.add_parser("parse", help="Extract content from a specific URL")
    p.add_argument("--url", "-u", required=True, help="URL to parse")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "search":
        print(google_search(query=args.query))
    elif args.command == "parse":
        print(web_parse(url=args.url))


if __name__ == "__main__":
    main()
