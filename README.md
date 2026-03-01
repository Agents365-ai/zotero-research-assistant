# Zotero Research Assistant

![Claude Code](https://img.shields.io/badge/Claude_Code-compatible-blue)
![OpenCode](https://img.shields.io/badge/OpenCode-compatible-green)

Semantic search over your Zotero library using **LanceDB** vector database with **local Qwen3** embedding and multiple reranking options optimized for Apple Silicon.

**No external servers required** - runs entirely on your local machine.

## Features

- **Semantic Search** - Find papers by meaning, not just keywords
- **Metadata Search** - Quick keyword search via Zotero API (no index needed)
- **Multiple Rerankers** - jina-v3 (MLX), Qwen3, or mmarco
- **Apple Silicon Optimized** - MLX native acceleration on M-series chips
- **Full-Text Indexing** - Extracts and indexes PDF content (first 10 pages)
- **Research Workspaces** - Create focused subsets for specific projects
- **Year Filtering** - Filter results by publication year
- **Incremental Sync** - Add new papers without rebuilding entire index
- **API Architecture** - Models loaded once, fast repeated queries

## Architecture

```
Zotero 7/8 (localhost:23119)
  ├─ Metadata API → Paper metadata
  └─ PDF Files → PyMuPDF text extraction
                      ↓
         model_server.py (:8765)
         ┌─────────────────────────────┐
         │ POST /embed                 │
         │   Qwen3-Embedding-0.6B (MPS)│
         │   • last_token_pool         │
         │   • L2 normalize            │
         │   • 1024-dim vectors        │
         ├─────────────────────────────┤
         │ POST /rerank                │
         │   • jina-v3-mlx (MLX)       │
         │   • qwen (CPU)              │
         │   • mmarco (CPU)            │
         └─────────────────────────────┘
                      ↓
         LanceDB (~/.local/share/zotero-lance/)
                      ↓
         Top-K results with scores
```

## Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 8GB | 16GB |
| GPU | - | Apple Silicon |
| Storage | 5GB | 20GB+ |
| Python | 3.10+ | 3.11 |

### Software Dependencies

1. **Zotero 7 or 8** with local API enabled
   - Download: https://www.zotero.org/download/
   - Zotero must be running (serves API on `localhost:23119`)

2. **Conda/Mamba** (recommended for environment management)
   - Miniforge: https://github.com/conda-forge/miniforge

3. **PyTorch** with MPS support (Apple Silicon)

### Models (auto-downloaded)

| Model | Size | Purpose | Backend |
|-------|------|---------|---------|
| `Qwen/Qwen3-Embedding-0.6B` | ~1.2GB | Text to vectors | MPS |
| `jinaai/jina-reranker-v3-mlx` | ~1.2GB | Reranking (default) | MLX |
| `Qwen/Qwen3-Reranker-0.6B` | ~1.2GB | Reranking (optional) | CPU |
| `cross-encoder/mmarco-mMiniLMv2` | ~500MB | Reranking (optional) | CPU |

Models download automatically on first run to `~/.cache/huggingface/`.

## Installation

### Option A: pip install

```bash
git clone https://github.com/Agents365-ai/zotero-research-assistant.git
cd zotero-research-assistant
pip install -e ".[mlx]"
```

After install, use `zra` and `zra-server` commands directly.

### Option B: Claude Code skill

```bash
git clone https://github.com/Agents365-ai/zotero-research-assistant.git
cp -r zotero-research-assistant ~/.claude/skills/
```

### Environment setup (if not using pip install)

```bash
conda create -n zotero-ra python=3.11
conda activate zotero-ra
pip install torch transformers lancedb pymupdf requests tqdm pyarrow flask sentence-transformers mlx mlx-lm huggingface_hub
```

## Quick Start

### 1. Start the Model Server

```bash
# Default: jina-v3-mlx (fastest on Apple Silicon)
zra-server --preload --port 8765

# Or with alternative rerankers:
zra-server --preload --reranker qwen   # Qwen3-Reranker
zra-server --preload --reranker mmarco # mmarco-mMiniLMv2
```

### 2. Build the Index

```bash
zra build              # Full library
zra build --limit 100  # First 100 papers (testing)
```

### 3. Search

```bash
zra search "single cell RNA sequencing" --rerank
```

## Usage

```bash
zra <command> [options]
```

### Search Commands

| Command | Needs Model Server | Description |
|---------|-------------------|-------------|
| `search "query" [--rerank] [-k N] [--year Y]` | Yes | Semantic search over indexed papers |
| `search "query" --add-to WS` | Yes | Search and add results to workspace |
| `meta-search "query" [--limit N]` | No | Zotero metadata search (title/author/tag) |

### Browse Commands

| Command | Needs Model Server | Description |
|---------|-------------------|-------------|
| `get KEY` | No | Full paper metadata from Zotero API |
| `fulltext KEY` | No | Zotero's indexed full-text content |
| `list [--limit N] [--offset N] [--sort year\|title]` | No | Browse indexed papers |
| `collections` | No | List Zotero collections |
| `tags [--limit N]` | No | List all Zotero tags |
| `status` | No | Index statistics |
| `delete [--force]` | No | Delete all data |

### Workspace Commands

| Command | Needs Model Server | Description |
|---------|-------------------|-------------|
| `ws-create NAME` | No | Create a new workspace |
| `ws-list` | No | List all workspaces |
| `ws-show NAME` | No | Show papers in workspace |
| `ws-add NAME KEYS` | No | Add papers (comma-separated keys) |
| `ws-remove NAME KEYS` | No | Remove papers |
| `ws-import NAME COLLECTION` | No | Import Zotero collection |
| `ws-search NAME "query" [--rerank] [-k N]` | Yes | Search within workspace |
| `ws-delete NAME` | No | Delete workspace |

### Interactive Shell

```bash
zra shell
```

## Model Server API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, shows reranker type |
| `/embed` | POST | `{"texts": [...], "mode": "query"\|"document"}` |
| `/rerank` | POST | `{"query": "...", "docs": [...], "top_k": 10}` |

```bash
# Health check
curl http://127.0.0.1:8765/health

# Embed texts
curl -X POST http://127.0.0.1:8765/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world"], "mode": "query"}'

# Rerank documents
curl -X POST http://127.0.0.1:8765/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "docs": ["doc1", "doc2"], "top_k": 2}'
```

## Reranker Comparison

| Reranker | Backend | Speed | Memory | Multilingual |
|----------|---------|-------|--------|--------------|
| **jina** (default) | MLX | ~0.5s | ~1.5GB | 26+ languages |
| **qwen** | CPU | ~2s | ~1.5GB | Chinese/English |
| **mmarco** | CPU | ~1s | ~500MB | 14 languages |

## Storage

```
~/.local/share/zotero-lance/
├── papers.lance/       # LanceDB vector database
├── meta.json           # Index metadata
├── config.json         # Configuration
└── workspaces.json     # Workspace definitions
```

## Troubleshooting

### Model server not responding
```bash
curl http://127.0.0.1:8765/health
pkill -f model_server.py
zra-server --preload --port 8765
```

### Zotero not connected
- Ensure Zotero is running
- Check API: `curl http://localhost:23119/api/users/0/items?limit=1`

### Out of memory
- Use `--reranker mmarco` (smaller model)
- Close other applications

## License

MIT
