---
name: zotero-research-assistant
description: Use when user wants to search Zotero library, query literature metadata, ask questions about papers, manage research workspaces, or build RAG indexes over their Zotero collection
---

# Zotero Research Assistant

Query your local Zotero 8 library with multimodal retrieval.

## Setup

- Python env: `/Users/niehu/mambaforge/envs/zotero-ra/bin/python`
- Zotero API: `http://localhost:23119` (must be running)
- Data: `~/.local/share/zotero-qdrant/` (v2) or `~/.local/share/zotero-research-assistant/` (v1)

## Quick Reference (Qdrant + Jina-v4)

| Task | Command |
|------|---------|
| Build index | `python workspace.py build` |
| Build (limit) | `python workspace.py build --limit 100` |
| Sync new papers | `python workspace.py sync` |
| Search | `python workspace.py search "query" -k 10` |
| Search (year filter) | `python workspace.py search "query" --year 2020` |
| Index status | `python workspace.py status` |
| Delete index | `python workspace.py delete` |

## Metadata Query

| Task | Command |
|------|---------|
| Search papers | `python zotero-query.py search "keyword"` |
| Paper details | `python zotero-query.py detail <key>` |
| List collections | `python zotero-query.py collections` |

## Legacy (UltraRAG - workspace_v1_ultrarag.py)

| Task | Command |
|------|---------|
| Build RAG (v1) | `python workspace_v1_ultrarag.py build` |
| Ask full library (v1) | `python workspace_v1_ultrarag.py qa "question"` |

All commands use: `/Users/niehu/mambaforge/envs/zotero-ra/bin/python /Users/niehu/.claude/skills/zotero-research-assistant/<script>`

## Workflow

1. **Build index**: `workspace.py build` - indexes all Zotero PDFs as page images
2. **Search**: `workspace.py search "query"` - two-stage retrieval (dense + MaxSim)
3. **Sync**: `workspace.py sync` - add new papers incrementally

## Architecture

```
Zotero 8 (localhost:23119)
  ├─ Metadata → zotero-query.py
  └─ PDFs → Page Images (PyMuPDF)
                ↓
         Jina-embeddings-v4
         ├─ Dense: 2048d (fast retrieval)
         └─ Multi: 128d × N tokens (MaxSim rerank)
                ↓
         Qdrant (local vector DB)
         ├─ Stage 1: Dense prefetch (top 100)
         └─ Stage 2: MaxSim rerank (top K)
                ↓
         Claude → answer with citations
```

## Models

- **V2**: `jinaai/jina-embeddings-v4` (multimodal, 4B params)
- **V1**: `openbmb/MiniCPM-Embedding-Light` + `jinaai/jina-embeddings-v4`

## Storage

- V2 index: `~/.local/share/zotero-qdrant/`
- V1 index: `~/.local/share/zotero-research-assistant/`
