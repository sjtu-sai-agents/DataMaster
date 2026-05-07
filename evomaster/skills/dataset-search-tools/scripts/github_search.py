#!/usr/bin/env python3
"""GitHub search CLI - wraps search_github.py

Usage:
    python github_search.py comprehensive --keyword "benchmark dataset" --search-type all
    python github_search.py repos  --keyword "nlp dataset" --sort stars
    python github_search.py code   --keyword "load_dataset" --language python
    python github_search.py issues --keyword "dataset release" --state open
    python github_search.py prs    --keyword "add dataset"
    python github_search.py users  --keyword "researcher"
    python github_search.py readme --owner awesomedata --repo awesome-public-datasets
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.search_github import (
    comprehensive_github_search,
    search_repositories,
    search_code,
    search_issues,
    search_pull_requests,
    search_users,
    get_repository_readme,
)


def main():
    parser = argparse.ArgumentParser(description="GitHub search CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("comprehensive", help="Search across repos, code, users, issues, PRs")
    p.add_argument("--keyword", "-k", required=True, help="Search keyword")
    p.add_argument("--search-type", "-t", default="all",
                   choices=["all", "repos", "code", "users", "issues", "prs"])
    p.add_argument("--user", default=None, help="Filter by user/org")
    p.add_argument("--language", default=None, help="Filter by language")
    p.add_argument("--per-page", type=int, default=10, help="Results per page (default: 10)")

    p = sub.add_parser("repos", help="Search repositories")
    p.add_argument("--keyword", "-k", required=True)
    p.add_argument("--user", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--sort", default=None, choices=["stars", "forks", "updated"])
    p.add_argument("--order", default=None, choices=["asc", "desc"])
    p.add_argument("--per-page", type=int, default=20)

    p = sub.add_parser("code", help="Search code")
    p.add_argument("--keyword", "-k", required=True)
    p.add_argument("--repo", default=None, help="Specific repo (owner/repo)")
    p.add_argument("--user", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--path", default=None, help="Filter by file path")
    p.add_argument("--per-page", type=int, default=20)

    p = sub.add_parser("issues", help="Search issues")
    p.add_argument("--keyword", "-k", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--state", default=None, choices=["open", "closed"])
    p.add_argument("--per-page", type=int, default=20)

    p = sub.add_parser("prs", help="Search pull requests")
    p.add_argument("--keyword", "-k", required=True)
    p.add_argument("--repo", default=None)
    p.add_argument("--user", default=None)
    p.add_argument("--state", default=None, choices=["open", "closed"])
    p.add_argument("--per-page", type=int, default=20)

    p = sub.add_parser("users", help="Search users")
    p.add_argument("--keyword", "-k", required=True)
    p.add_argument("--per-page", type=int, default=20)

    p = sub.add_parser("readme", help="Get repository README")
    p.add_argument("--owner", required=True, help="Repository owner")
    p.add_argument("--repo", required=True, help="Repository name")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "comprehensive":
        print(comprehensive_github_search(
            keyword=args.keyword, search_type=args.search_type,
            user=args.user, language=args.language, per_page=args.per_page,
        ))
    elif args.command == "repos":
        print(search_repositories(
            keyword=args.keyword, user=args.user, language=args.language,
            sort=args.sort, order=args.order, per_page=args.per_page,
        ))
    elif args.command == "code":
        print(search_code(
            keyword=args.keyword, repo=args.repo, user=args.user,
            language=args.language, path=args.path, per_page=args.per_page,
        ))
    elif args.command == "issues":
        print(search_issues(
            keyword=args.keyword, repo=args.repo, user=args.user,
            state=args.state, per_page=args.per_page,
        ))
    elif args.command == "prs":
        print(search_pull_requests(
            keyword=args.keyword, repo=args.repo, user=args.user,
            state=args.state, per_page=args.per_page,
        ))
    elif args.command == "users":
        print(search_users(keyword=args.keyword, per_page=args.per_page))
    elif args.command == "readme":
        print(get_repository_readme(owner=args.owner, repo=args.repo))


if __name__ == "__main__":
    main()
