---
name: zotero-research-assistant
description: Use when user wants to search Zotero library, query literature metadata, ask questions about papers, manage research workspaces, or build RAG indexes over their Zotero collection
---

# Zotero Research Assistant

Semantic search over your Zotero library using LanceDB + local Qwen3 models.

## Setup

- Python env: `/Users/niehu/mambaforge/envs/zotero-ra/bin/python`
- Zotero API: `http://localhost:23119` (must be running)
- Data: `~/.local/share/zotero-lance/`

## Quick Reference

| Task | Command |
|------|---------|
| Build index | `python workspace.py build` |
| Build (limit) | `python workspace.py build --limit 100` |
| Sync new papers | `python workspace.py sync` |
| Search | `python workspace.py search "query"` |
| Search (top K) | `python workspace.py search "query" -k 20` |
| Search (year filter) | `python workspace.py search "query" --year 2020` |
| Index status | `python workspace.py status` |
| Delete index | `python workspace.py delete` |

## Metadata Query

| Task | Command |
|------|---------|
| Search papers | `python zotero-query.py search "keyword"` |
| Paper details | `python zotero-query.py detail <key>` |
| List collections | `python zotero-query.py collections` |

All commands use: `/Users/niehu/mambaforge/envs/zotero-ra/bin/python /Users/niehu/.claude/skills/zotero-research-assistant/<script>`

## Architecture

```
Zotero 7/8 (localhost:23119)
  ├─ Metadata → zotero-query.py
  └─ PDFs → PyMuPDF text extraction
               ↓
       Qwen3-Embedding-0.6B (local)
               ↓
       LanceDB (similarity search)
               ↓
       Qwen3-Reranker-0.6B (local)
               ↓
       Top-K results with scores
```

## Models (Local, No External Server)

| Model | Purpose | Size |
|-------|---------|------|
| Qwen/Qwen3-Embedding-0.6B | Text to vectors | ~1.2GB |
| Qwen/Qwen3-Reranker-0.6B | Score query-doc pairs | ~1.2GB |

Models auto-download on first run. No Ollama or LM Studio required.

## Performance

- **Build speed**: ~2-3 papers/minute
- **Search latency**: 1-3 seconds (includes reranking)
- **Storage**: ~4MB per paper

## Storage

Index location: `~/.local/share/zotero-lance/`
