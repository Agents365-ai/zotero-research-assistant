---
name: zotero-research-assistant
description: Use when user wants to search Zotero library, query literature metadata, ask questions about papers, manage research workspaces, or build RAG indexes over their Zotero collection
---

# Zotero Research Assistant

## Setup

- Python: `/Users/niehu/mambaforge/envs/zotero-ra/bin/python`
- Scripts: `/Users/niehu/.claude/skills/zotero-research-assistant/`
- Run commands as: `$PYTHON $SCRIPTS/workspace.py <command>`

## Pre-flight (before ANY command that needs embedding/reranking)

Check model server and Zotero are running:

```bash
curl -sf http://127.0.0.1:8765/health && echo "Model server OK" || echo "Model server DOWN"
curl -sf http://localhost:23119/api/users/0/items?limit=1 > /dev/null && echo "Zotero OK" || echo "Zotero DOWN"
```

- Model server down → tell user: `python model_server.py --preload --port 8765`
- Zotero down → tell user to open Zotero app

Note: `meta-search`, `get`, `tags`, `collections` only need Zotero, not model server.

## Workflows

### "Find papers about X" (semantic search)
1. Pre-flight check
2. `workspace.py search "<query>" --rerank -k 10`
3. Format results as readable table (title, authors, year, score)
4. Offer: "Want to add these to a workspace?"
5. If yes → re-run with `--add-to <workspace_name>`

### "Search for papers by keyword/author/tag" (metadata search)
1. `workspace.py meta-search "<keyword>" --limit 20`
2. No model server needed — queries Zotero API directly

### "Tell me about paper X" / "Get details for KEY"
1. `workspace.py get <key>`
2. Returns full metadata (title, authors, DOI, tags, PDF paths)

### "What's in my library about X?" (first time)
1. Check if index exists: `workspace.py status`
2. If no index → `workspace.py build` (full) or `workspace.py build --limit 100` (quick test)
3. Then search as above

### "Sync / update my index"
1. `workspace.py sync`
2. Reports count of new papers added

### "Create a workspace for project X"
1. `workspace.py ws-create "<name>"`
2. Search for relevant papers: `workspace.py search "<topic>" --rerank -k 20`
3. Add results: `workspace.py ws-add "<name>" "<key1>,<key2>,..."`
4. Or import a Zotero collection: `workspace.py ws-import "<name>" "<collection>"`

### "Search within my workspace"
1. `workspace.py ws-search "<name>" "<query>" --rerank`

### "Show me all tags" / "List collections"
1. `workspace.py tags`
2. `workspace.py collections`

## Command Reference

| Command | Needs Model Server | Description |
|---------|-------------------|-------------|
| `search "q" [--rerank] [-k N] [--year Y]` | Yes | Semantic search |
| `meta-search "q" [--limit N]` | No | Zotero metadata search |
| `get <key>` | No | Paper details |
| `fulltext <key>` | No | Zotero full-text content |
| `tags [--limit N]` | No | List all tags |
| `collections` | No | List Zotero collections |
| `build [--limit N] [--collection C]` | Yes | Build vector index |
| `sync` | Yes | Add new papers to index |
| `list [--limit N] [--offset N]` | No | Browse indexed papers |
| `status` | No | Index statistics |
| `ws-create <name>` | No | Create workspace |
| `ws-list` | No | List workspaces |
| `ws-show <name>` | No | Show workspace papers |
| `ws-add <name> <keys>` | No | Add papers to workspace |
| `ws-remove <name> <keys>` | No | Remove papers |
| `ws-import <name> <collection>` | No | Import collection |
| `ws-search <name> "q" [--rerank]` | Yes | Search within workspace |
| `ws-delete <name>` | No | Delete workspace |
| `delete [--force]` | No | Delete all data |
