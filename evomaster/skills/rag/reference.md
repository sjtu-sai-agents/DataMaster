# RAG Reference Guide

## Architecture Overview

### Vector Database Structure

The actual database is located at `evomaster/skills/rag/MLE_DATABASE/`:

```
MLE_DATABASE/
├── MLE75_tasksummary_v2.json          # Task summaries JSON file
├── simple_instructions/                # Simple instructions directory
│   ├── <task_name>/
│   │   └── simple_instructions.txt
│   └── ...
└── node_vectorstore/                  # Vector stores (default: 768 dims)
    ├── draft/                          # Draft stage vectorstore
    │   ├── faiss.index                 # FAISS index file (binary)
    │   ├── embeddings.npy              # Pre-computed embeddings (numpy array)
    │   ├── nodes.jsonl                 # Node mappings (one JSON per line)
    │   └── draft_407_75_db.json        # Full node data with knowledge content
    ├── improve/                        # Improve stage vectorstore
    │   ├── faiss.index
    │   ├── embeddings.npy
    │   ├── nodes.jsonl
    │   └── nodes_for_improve.json
    └── debug/                          # Debug stage vectorstore
        ├── faiss.index
        ├── embeddings.npy
        ├── nodes.jsonl
        └── nodes_for_debug.json
```

**Note**: Multiple vectorstore directories exist for different embedding dimensions. Each directory corresponds to a specific model:

- `node_vectorstore/` (768 dims) - Uses local model: `evomaster/skills/rag/local_models/all-mpnet-base-v2`
- `node_vectorstore_512/` (512 dims) - Uses `text-embedding-3-large` (512 dimensions)
- `node_vectorstore_768/` (768 dims) - Uses `text-embedding-3-large` (768 dimensions)
- `node_vectorstore_1024/` (1024 dims) - Uses `text-embedding-3-large` (1024 dimensions)
- `node_vectorstore_2048/` (2048 dims) - Uses `text-embedding-3-large` (2048 dimensions)
- `node_vectorstore_3072/` (3072 dims) - Uses `text-embedding-3-large` (3072 dimensions, default)

**Important**: Use the vectorstore directory that matches your model's output dimension. The default `node_vectorstore/` uses the local model, while other directories use OpenAI's `text-embedding-3-large` model with different dimensions.

### File Formats

#### nodes.jsonl

Format varies by stage:

**Draft stage** (`node_vectorstore/draft/nodes.jsonl`):
```json
{"task_name": "aerial-cactus-identification", "summary": "This is a binary image classification task..."}
{"task_name": "cassava-leaf-disease-classification", "summary": "This is a multiclass image classification task..."}
```

**Improve/Debug stages** (`node_vectorstore/improve/nodes.jsonl`):
```json
{"node_id": "552dbb3683904474a8dff7ee370e472a", "parent_full_plan": "The pipeline initializes..."}
{"node_id": "47907dcc2fbe4a089c457b6790cd7f9b", "parent_full_plan": "..."}
```

**Note**: The script automatically handles missing `node_id` fields by using `task_name` or line index as fallback.

#### nodes_data.json

Format varies by stage:

**Draft stage** (`draft_407_75_db.json`):
```json
{
  "aerial-cactus-identification": {
    "summary": "This is a binary image classification task...",
    "data_knowledge": "The code implements a PyTorch-based data loading...",
    "model_knowledge": "Model Selection: Uses a pretrained ResNet18..."
  },
  "cassava-leaf-disease-classification": {
    "summary": "...",
    "data_knowledge": "...",
    "model_knowledge": "..."
  }
}
```

**Improve/Debug stages** (`nodes_for_improve.json`, `nodes_for_debug.json`):
```json
{
  "552dbb3683904474a8dff7ee370e472a": {
    "meta_info": {
      "task_name": "...",
      "timestamp": "2024-01-01T00:00:00"
    },
    "content": {
      "improve_knowledge": ["Knowledge point 1", "Knowledge point 2"],
      "full_plan": "...",
      "code": "..."
    },
    "metric": "..."
  }
}
```

**Notes**:
- The schema of `nodes_data.json` is **not strictly constrained**; you can customize fields according to your business needs.
- `scripts/search.py` supports `--content_path` to specify fields to retrieve using dot notation (e.g., `content.text`, `content.code`, `meta_info.task_name`, `data_knowledge`, `model_knowledge`).

## Embedding Models

### Default Model

**Local model** (default, for `node_vectorstore/`): `evomaster/skills/rag/local_models/all-mpnet-base-v2` - 768 dimensions

**Important**: When using a local model path, ensure the model directory contains `config.json`, `model.safetensors`, and tokenizer files.

### Vectorstore-Model Mapping

Each vectorstore directory corresponds to a specific model:

| Vectorstore Directory | Model | Dimensions | Notes |
|----------------------|-------|------------|-------|
| `node_vectorstore/` | `evomaster/skills/rag/local_models/all-mpnet-base-v2` (local) | 768 | Default, uses local model |
| `node_vectorstore_512/` | `text-embedding-3-large` | 512 | OpenAI embedding model |
| `node_vectorstore_768/` | `text-embedding-3-large` | 768 | OpenAI embedding model |
| `node_vectorstore_1024/` | `text-embedding-3-large` | 1024 | OpenAI embedding model |
| `node_vectorstore_2048/` | `text-embedding-3-large` | 2048 | OpenAI embedding model |
| `node_vectorstore_3072/` | `text-embedding-3-large` | 3072 | OpenAI embedding model (default) |

### Recommended Models

| Model | Dimensions | Speed | Accuracy | Use Case | Vectorstore Directory |
|-------|------------|-------|----------|----------|----------------------|
| `evomaster/skills/rag/local_models/all-mpnet-base-v2` (local) | 768 | Medium | High | Default, balanced, offline use | `node_vectorstore/` |
| `text-embedding-3-large` (512-dim) | 512 | Medium | Very High | High accuracy, smaller size | `node_vectorstore_512/` |
| `text-embedding-3-large` (768-dim) | 768 | Medium | Very High | High accuracy | `node_vectorstore_768/` |
| `text-embedding-3-large` (1024-dim) | 1024 | Medium | Very High | High accuracy | `node_vectorstore_1024/` |
| `text-embedding-3-large` (2048-dim) | 2048 | Medium | Very High | High accuracy | `node_vectorstore_2048/` |
| `text-embedding-3-large` (3072-dim) | 3072 | Medium | Very High | Highest accuracy (default) | `node_vectorstore_3072/` |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Medium | High | HuggingFace, balanced | Custom |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast | Medium | Speed priority | Custom |

### Model Loading

- **Local models**: Loaded directly from the specified path (fast, no download needed)
  - Example: `evomaster/skills/rag/local_models/all-mpnet-base-v2` used with `node_vectorstore/`
- **OpenAI models** (`text-embedding-3-large`): Requires API access, used with dimension-specific vectorstore directories
  - 512-dim version → `node_vectorstore_512/`
  - 768-dim version → `node_vectorstore_768/`
  - 1024-dim version → `node_vectorstore_1024/`
  - 2048-dim version → `node_vectorstore_2048/`
  - 3072-dim version (default) → `node_vectorstore_3072/`
- **HuggingFace models**: Automatically downloaded from HuggingFace on first use. Ensure you have internet access or pre-download models.

**Important**: Always use the vectorstore directory that matches the embedding model and dimension used during indexing. Mixing incompatible models and vectorstores will result in incorrect search results.

## FAISS Index Types

### Index Types

- **IndexFlatL2**: Exact search, slow for large datasets
- **IndexIVFFlat**: Approximate search, faster, requires training
- **IndexHNSW**: Hierarchical navigable small world, fast approximate search

### Distance Metrics

- **L2 (Euclidean)**: Default, lower is better
- **Inner Product**: For normalized vectors, higher is better
- **Cosine**: Similar to inner product for normalized vectors

## Usage Examples

### Example 1: Basic Retrieval

Using `use_skill` tool:
```text
use_skill(
  skill_name="rag",
  action="run_script",
  script_name="search.py",
  script_args="--vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json --query \"How to improve accuracy?\" --top_k 5 --output json"
)
```

Direct command line execution (equivalent):
```bash
python evomaster/skills/rag/scripts/search.py \
  --vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft \
  --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json \
  --query "How to improve accuracy?" \
  --top_k 5 \
  --output json
```

### Example 2: Task Knowledge (basic tools + search.py)

1. Get the query text: use `execute_bash` to `cat evomaster/skills/rag/MLE_DATABASE/simple_instructions/<task_name>/simple_instructions.txt` from project root, or use `str_replace_editor` (command `view`) with the absolute path to that file.
2. Call `search.py` with that query:

```text
use_skill(
  skill_name="rag",
  action="run_script",
  script_name="search.py",
  script_args="--vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json --query \"<query from step 1>\" --top_k 1 --threshold 1.5 --model evomaster/skills/rag/local_models/all-mpnet-base-v2 --output json"
)
```

3. From the JSON `results[0].content` (or the node in nodes_data for that `node_id`), extract `data_knowledge` and `model_knowledge`.

### Example 3: Custom Knowledge Extraction

It is not recommended to extend the skill as an "importable Python package"; it is better to use script parameterization:

- Retrieve `data_knowledge` from draft stage:
```bash
python evomaster/skills/rag/scripts/search.py \
  --vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft \
  --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json \
  --query "..." \
  --content_path data_knowledge \
  --output json
```

- Retrieve `model_knowledge` from draft stage:
```bash
python evomaster/skills/rag/scripts/search.py \
  --vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft \
  --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json \
  --query "..." \
  --content_path model_knowledge \
  --output json
```

- Retrieve `improve_knowledge` from improve stage:
```bash
python evomaster/skills/rag/scripts/search.py \
  --vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/improve \
  --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/improve/nodes_for_improve.json \
  --query "..." \
  --content_path content.improve_knowledge \
  --output json
```

### Example 4: Text Encoding

Using `use_skill` tool:
```text
use_skill(
  skill_name="rag",
  action="run_script",
  script_name="encode.py",
  script_args="--text \"What is the main topic?\" --model evomaster/skills/rag/local_models/all-mpnet-base-v2"
)
```

Direct command line execution:
```bash
python evomaster/skills/rag/scripts/encode.py \
  --text "What is the main topic?" \
  --model evomaster/skills/rag/local_models/all-mpnet-base-v2
```

## Performance Optimization

### Speed Optimization

1. **Use local models**: Local models load faster than downloading from HuggingFace
2. **Use smaller models**: `all-MiniLM-L6-v2` is ~3x faster than `all-mpnet-base-v2`
3. **GPU acceleration**: Set `device="cuda"` if available (currently scripts default to CPU)
4. **Index optimization**: Use IndexIVFFlat or IndexHNSW for large datasets

### Memory Optimization

1. **CPU mode**: Use `device="cpu"` to reduce memory usage (default)
2. **Lazy loading**: Load nodes_data only when needed

## Integration Patterns

### Pattern 1: Multi-Stage RAG

Using `use_skill` tool for different stages:

```text
# Draft stage
use_skill(
  skill_name="rag",
  action="run_script",
  script_name="search.py",
  script_args="--vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json --query \"...\" --top_k 5"
)

# Improve stage
use_skill(
  skill_name="rag",
  action="run_script",
  script_name="search.py",
  script_args="--vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/improve --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/improve/nodes_for_improve.json --query \"...\" --top_k 5"
)

# Debug stage
use_skill(
  skill_name="rag",
  action="run_script",
  script_name="search.py",
  script_args="--vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/debug --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/debug/nodes_for_debug.json --query \"...\" --top_k 5"
)
```

### Pattern 2: Task Knowledge (basic tools + search.py)

1. Read query from `evomaster/skills/rag/MLE_DATABASE/simple_instructions/<task_name>/simple_instructions.txt` using `execute_bash` (cat from project root) or `str_replace_editor` (view).
2. Call `search.py` with that query and `--output json`; extract `data_knowledge` and `model_knowledge` from `results[0].content`.

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: 
   - Check that all required files exist in vec_dir
   - Ensure paths are relative to project root (starting with `evomaster/`) or absolute
   - Script automatically resolves paths relative to project root

2. **CUDA out of memory**: 
   - Scripts default to CPU mode
   - If using GPU, reduce batch_size or use CPU mode

3. **Poor retrieval quality**: 
   - Check embedding model quality
   - Adjust distance thresholds
   - Verify data quality in nodes_data.json
   - Ensure using the correct vectorstore directory matching your model's dimension

4. **Slow search**: 
   - Use smaller embedding models
   - Use local models instead of downloading from HuggingFace
   - Optimize FAISS index type
   - Use GPU if available

5. **Path resolution errors**:
   - Ensure script can find project root (contains `evomaster/` directory)
   - Use relative paths starting with `evomaster/` for consistency
   - Script automatically resolves paths relative to project root

### Debugging

Enable debug logging by modifying script or setting environment variable:
```bash
export PYTHONPATH=evomaster/skills/rag/scripts:$PYTHONPATH
python evomaster/skills/rag/scripts/search.py --vec_dir ... --query "..." --top_k 100
```

Check distance distributions in script output (distances are included in JSON results).

## Migration from agentic4mle

The RAG skill is designed to be compatible with agentic4mle's vectorstore structure:

1. **Same file structure**: Uses the same directory layout
2. **Same data format**: Compatible with nodes.jsonl and nodes_data.json
3. **Script-based interface**: Scripts provide similar functionality to BaseMemorySearcher

To migrate, use the scripts via `use_skill` tool or direct command line:

```bash
# Draft stage
python evomaster/skills/rag/scripts/search.py \
  --vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft \
  --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/draft/draft_407_75_db.json \
  --query "..." \
  --top_k 5 \
  --output json

# Improve stage
python evomaster/skills/rag/scripts/search.py \
  --vec_dir evomaster/skills/rag/MLE_DATABASE/node_vectorstore/improve \
  --nodes_data evomaster/skills/rag/MLE_DATABASE/node_vectorstore/improve/nodes_for_improve.json \
  --query "..." \
  --top_k 5 \
  --output json
```

## Best Practices

1. **Consistent model**: Use the same embedding model for indexing and querying
2. **Distance thresholds**: Set appropriate thresholds based on your data (default: 1.5)
3. **Metadata**: Include rich metadata in nodes_data.json for filtering
4. **Path consistency**: Use relative paths starting with `evomaster/` for portability
5. **Model selection**: Prefer local models for faster loading and offline use
6. **Evaluation**: Regularly evaluate retrieval quality by checking distance distributions

## Future Enhancements

Planned features:

- [ ] Database building implementation
- [ ] Support for more vector databases (Pinecone, Weaviate, etc.)
- [ ] Hybrid search (dense + sparse)
- [ ] Reranking support
- [ ] Metadata filtering
- [ ] Incremental updates
- [ ] GPU acceleration support in scripts
