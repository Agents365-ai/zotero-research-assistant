# Zotero Research Assistant

![Claude Code](https://img.shields.io/badge/Claude_Code-compatible-blue)
![OpenCode](https://img.shields.io/badge/OpenCode-compatible-green)

Semantic search over your Zotero library using **LanceDB** vector database with **Qwen3** embedding and reranking models.

## Features

- **Semantic Search** - Find papers by meaning, not just keywords
- **Neural Reranking** - Qwen3-Reranker-4B improves result relevance
- **Full-Text Indexing** - Extracts and indexes PDF content (first 10 pages)
- **Year Filtering** - Filter results by publication year
- **Incremental Sync** - Add new papers without rebuilding entire index

## Architecture

```
Zotero 7/8 (localhost:23119)
  ├─ Metadata API → Paper metadata
  └─ PDF Files → PyMuPDF text extraction
                      ↓
              Qwen3-Embedding-4B (4B params)
                      ↓
              LanceDB vector storage
                      ↓
              Vector similarity search
                      ↓
              Qwen3-Reranker-4B (reranking)
                      ↓
              Top-K results with scores
```

## Installation

### Requirements

- Python 3.10+
- [Zotero 7/8](https://www.zotero.org/) running locally (localhost:23119)
- ~16GB RAM (for Qwen3 4B models)
- Apple Silicon / NVIDIA GPU recommended

### Setup

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

### Commands

| Command | Description |
|---------|-------------|
| `build` | Build index from all Zotero PDFs |
| `build --limit N` | Build index from first N papers |
| `sync` | Add new papers to existing index |
| `search "query"` | Semantic search with reranking |
| `search "query" -k 20` | Return top 20 results |
| `search "query" --year 2020` | Filter by minimum year |
| `status` | Show index statistics |
| `delete` | Delete the index |

### Examples

```bash
# Build index (first 100 papers for testing)
python workspace.py build --limit 100

# Search
python workspace.py search "single cell RNA sequencing"

# Search with filters
python workspace.py search "CRISPR gene editing" -k 5 --year 2022

# Check index status
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
| Embedding model | Qwen3-Embedding-4B |
| Reranker model | Qwen3-Reranker-4B |
| Build speed | ~1-2 papers/minute |
| Search latency | 2-5 seconds (includes reranking) |
| Storage | ~4MB per paper |

## Storage

```
~/.local/share/zotero-lance/
├── papers.lance/    # LanceDB vector database
└── meta.json        # Index metadata
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
