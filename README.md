# Zotero Research Assistant

![Claude Code](https://img.shields.io/badge/Claude_Code-compatible-blue)
![OpenCode](https://img.shields.io/badge/OpenCode-compatible-green)

Semantic search over your Zotero library using **LanceDB** vector database with **local Qwen3** embedding and reranking models.

**No external servers required** - runs entirely on your local machine.

## Features

- **Semantic Search** - Find papers by meaning, not just keywords
- **Neural Reranking** - Qwen3-Reranker improves result relevance
- **Full-Text Indexing** - Extracts and indexes PDF content (first 10 pages)
- **Research Workspaces** - Create focused subsets for specific projects
- **Year Filtering** - Filter results by publication year
- **Incremental Sync** - Add new papers without rebuilding entire index
- **Zero Configuration** - No Ollama/LM Studio setup needed

## Architecture

```
Zotero 7/8 (localhost:23119)
  ├─ Metadata API → Paper metadata
  └─ PDF Files → PyMuPDF text extraction
                      ↓
              Qwen3-Embedding-0.6B (local)
                      ↓
              LanceDB vector storage
                      ↓
              Vector similarity search
                      ↓
              Qwen3-Reranker-0.6B (local)
                      ↓
              Top-K results with scores
```

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8GB | 16GB |
| GPU | - | Apple Silicon / NVIDIA |
| Storage | 5GB | 20GB+ |
| Python | 3.10+ | 3.11 |

### Software Dependencies

1. **Zotero 7 or 8** with local API enabled
   - Download: https://www.zotero.org/download/
   - Zotero must be running (serves API on `localhost:23119`)

2. **Conda/Mamba** (recommended for environment management)
   - Miniforge: https://github.com/conda-forge/miniforge

3. **PyTorch** with MPS/CUDA support
   - Apple Silicon: MPS backend (automatic)
   - NVIDIA: CUDA toolkit required

### Hugging Face Models (auto-downloaded)

| Model | Size | Purpose |
|-------|------|---------|
| `Qwen/Qwen3-Embedding-0.6B` | ~1.2GB | Text to vectors |
| `Qwen/Qwen3-Reranker-0.6B` | ~1.2GB | Query-document scoring |

Models download automatically on first run to `~/.cache/huggingface/`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Agents365-ai/zotero-research-assistant.git
```

2. Install as Claude Code skill:
```bash
cp -r zotero-research-assistant ~/.claude/skills/
```

Or for OpenCode:
```bash
cp -r zotero-research-assistant ~/.opencode/skills/
```

3. Create conda environment:
```bash
conda create -n zotero-ra python=3.10
conda activate zotero-ra
pip install torch transformers lancedb pymupdf requests tqdm pyarrow
```

## Usage

```bash
python workspace.py <command> [options]
```

### Index Commands

| Command | Description |
|---------|-------------|
| `build` | Build index from all Zotero PDFs |
| `build --limit N` | Build index from first N papers |
| `build --collection NAME` | Build index from specific collection |
| `sync` | Add new papers to existing index |
| `search "query"` | Semantic search |
| `search "query" -k 20` | Return top 20 results |
| `search "query" --year 2020` | Filter by minimum year |
| `search "query" --add-to WS` | Add results to workspace |
| `collections` | List Zotero collections |
| `status` | Show index statistics |
| `delete` | Delete the index |

### Workspace Commands

Workspaces let you create focused paper subsets for specific research projects.

| Command | Description |
|---------|-------------|
| `ws-list` | List all workspaces |
| `ws-create NAME` | Create a new workspace |
| `ws-add NAME KEYS` | Add papers by key (comma-separated) |
| `ws-import NAME COLLECTION` | Import Zotero collection to workspace |
| `ws-search NAME "query"` | Search within workspace (with reranking) |
| `ws-delete NAME` | Delete a workspace |

### Examples

**Natural language prompts** (when using with AI coding assistant):

```
"Build the Zotero search index"
"Search my library for papers about single cell RNA sequencing"
"Find recent papers on CRISPR gene editing from 2022 onwards"
"Create a workspace called 'scRNA-seq' for my single cell papers"
"Import the 'Machine Learning' collection to a workspace"
"Search my ML workspace for transformer architectures"
```

**Direct CLI commands**:

```bash
# Build index (first 100 papers for testing)
python workspace.py build --limit 100

# Build from specific collection
python workspace.py build --collection "Machine Learning"

# Search
python workspace.py search "single cell RNA sequencing"

# Search with filters
python workspace.py search "CRISPR gene editing" -k 5 --year 2022

# Create workspace and add search results
python workspace.py ws-create myproject
python workspace.py search "deep learning" --add-to myproject

# Import collection to workspace
python workspace.py ws-import myproject "Deep Learning Papers"

# Search within workspace (uses reranking)
python workspace.py ws-search myproject "attention mechanism"

# Check status
python workspace.py status
```

## Metadata Queries

Use `zotero-query.py` for metadata operations (no vector search):

```bash
python zotero-query.py search "keyword"    # Search by metadata
python zotero-query.py detail <key>        # Get paper details
python zotero-query.py collections         # List collections
```

## Performance

| Metric | Value |
|--------|-------|
| Embedding model | Qwen3-Embedding-0.6B |
| Reranker model | Qwen3-Reranker-0.6B |
| Build speed | ~2-3 papers/minute |
| Search latency | 1-3 seconds |
| Workspace search | 2-5 seconds (includes reranking) |
| Storage | ~4MB per paper |

## Storage

```
~/.local/share/zotero-lance/
├── papers.lance/       # LanceDB vector database
├── meta.json           # Index metadata
├── config.json         # Configuration
└── workspaces.json     # Workspace definitions
```

## Dependencies

```
torch
transformers
lancedb
pymupdf
requests
tqdm
pyarrow
```

## License

MIT
