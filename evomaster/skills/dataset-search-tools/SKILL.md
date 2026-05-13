---
name: dataset-search-tools
description: Dataset discovery and download toolkit covering HuggingFace, GitHub, Google, and academic paper search. Use when you need to find, inspect, or download ML datasets.
license: Complete terms in LICENSE.txt
---

# Dataset Search Tools

Search, inspect, and download datasets from HuggingFace Hub, GitHub, Google, and academic papers. All operations available as CLI scripts under `scripts/`. Source implementations copied from `playground/search_dataset_tools/`.

Download target: `{workspace}/data_links/`

## HuggingFace

Primary source for ML datasets. HuggingFace search is **keyword-exact**, not semantic.

Key concepts: `dataset_id` (`org/name`), `config` (subset/language), `split` (`train`/`validation`/`test`).

| Tool | CLI |
|------|-----|
| `search_datasets` | `python scripts/hf_search.py search -q "sentiment" -l 10` |
| `search_datasets` (author) | `python scripts/hf_search.py search -q "translation" -a Helsinki-NLP` |
| `inspect_dataset` | `python scripts/hf_search.py inspect --id stanfordnlp/sst2` |
| `inspect_dataset` (config) | `python scripts/hf_search.py inspect --id stanfordnlp/sst2 -c default` |
| `get_dataset_configs` | `python scripts/hf_search.py configs --id glue` |
| `get_dataset_splits` | `python scripts/hf_search.py splits --id stanfordnlp/sst2` |
| `get_dataset_sample` | `python scripts/hf_search.py sample --id stanfordnlp/sst2 -n 5 -s train` |
| `get_dataset_readme` | `python scripts/hf_search.py readme --id stanfordnlp/sst2` |
| `download_dataset` | `python scripts/hf_search.py download --id stanfordnlp/sst2 -o ./data_links/sst2` |

Typical workflow:
```bash
python scripts/hf_search.py search -q "question answering" -l 10
python scripts/hf_search.py inspect --id openai/gsm8k
python scripts/hf_search.py sample --id openai/gsm8k -n 3
python scripts/hf_search.py download --id openai/gsm8k -o ./data_links/gsm8k
```

## GitHub

Good for research datasets, benchmarks, curated collections.

| Tool | CLI |
|------|-----|
| `comprehensive_github_search` | `python scripts/github_search.py comprehensive -k "benchmark dataset" --search-type all` |
| `search_repositories` | `python scripts/github_search.py repos -k "nlp dataset" --sort stars` |
| `search_code` | `python scripts/github_search.py code -k "load_dataset" --language python` |
| `search_issues` | `python scripts/github_search.py issues -k "dataset release" --state open` |
| `search_pull_requests` | `python scripts/github_search.py prs -k "add dataset"` |
| `search_users` | `python scripts/github_search.py users -k "researcher"` |
| `get_repository_readme` | `python scripts/github_search.py readme --owner awesomedata --repo awesome-public-datasets` |

## Web Search

Workflow: `search` first to get URLs, then `parse` specific URLs.

| Tool | CLI |
|------|-----|
| `google_search` | `python scripts/web_search.py search -q "public NLP dataset download"` |
| `web_parse` | `python scripts/web_search.py parse -u "https://..."` |

## Academic Papers

After finding PDF URLs, use `web_parse` to read content.

| Tool | CLI |
|------|-----|
| `arxiv_search_by_content` | `python scripts/scholar_search.py arxiv -q "benchmark dataset" -n 10` |
| `arxiv_search_by_author` | `python scripts/scholar_search.py arxiv-author -a "Yann LeCun" -n 5` |
| `google_scholar_search` | `python scripts/scholar_search.py scholar -q "sentiment dataset"` |
| `search_dblp_papers` | `python scripts/scholar_search.py dblp-papers -q "Diffusion Model" -n 5` |
| `search_dblp_authors` | `python scripts/scholar_search.py dblp-authors -q "Geoffrey Hinton" -n 5` |
| `search_dblp_venues` | `python scripts/scholar_search.py dblp-venues -q "CVPR" -n 5` |

## Data Validation

```bash
python scripts/validate_dataset.py -p ./data_links/my_dataset
```

Checks file structure, CSV/JSON/text/parquet quality, missing values, duplicates, basic statistics.

## Scripts Structure

```
scripts/
├── github_search.py
├── hf_search.py
├── scholar_search.py
├── src
│   ├── search_github.py
│   ├── search_huggingface.py
│   ├── search_scholar.py
│   └── search_web.py
└── web_search.py
```

