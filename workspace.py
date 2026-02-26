#!/usr/bin/env python3
"""Zotero Research Assistant - LanceDB + Local Qwen3 Models.

Usage:
  python workspace.py build [--collection NAME]  Build index (all or collection)
  python workspace.py sync                       Sync new papers
  python workspace.py search <query> [-k] [--add-to WS]  Search (optionally add to workspace)

  # Workspace commands
  python workspace.py ws-list                    List workspaces
  python workspace.py ws-create <name>           Create workspace
  python workspace.py ws-add <name> <keys>       Add papers to workspace
  python workspace.py ws-import <name> <coll>    Import collection to workspace
  python workspace.py ws-search <name> <query>   Search within workspace
  python workspace.py ws-delete <name>           Delete workspace

  # Utility
  python workspace.py collections                List Zotero collections
  python workspace.py status                     Show status
  python workspace.py delete                     Delete all data
"""
import sys, os, json, argparse, re, time
from pathlib import Path
from typing import Optional, List, Dict
import urllib.parse

DATA_DIR = Path.home() / ".local/share/zotero-lance"
CONFIG_FILE = DATA_DIR / "config.json"
WORKSPACES_FILE = DATA_DIR / "workspaces.json"
TABLE_NAME = "papers"

# Model configuration
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"

def out(msg): print(f"[zotero] {msg}", flush=True)

# --- Config Management ---

def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}

def save_config(config):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

def load_workspaces():
    if WORKSPACES_FILE.exists():
        return json.loads(WORKSPACES_FILE.read_text())
    return {}

def save_workspaces(ws):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACES_FILE.write_text(json.dumps(ws, indent=2))

# --- Local Embedding Model ---

_embed_model = None
_embed_tokenizer = None
_embed_device = None
_embed_dim = None

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_embed_model():
    global _embed_model, _embed_tokenizer, _embed_device, _embed_dim
    if _embed_model is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        out(f"Loading embedding model: {EMBED_MODEL}")
        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        _embed_device = get_device()
        _embed_model = _embed_model.to(_embed_device).half()
        _embed_dim = _embed_model.config.hidden_size
        out(f"Embedding model loaded on {_embed_device} (dim={_embed_dim})")
    return _embed_model, _embed_tokenizer

def get_embeddings(texts: List[str], mode: str = "document") -> List[List[float]]:
    import torch
    model, tokenizer = get_embed_model()

    if hasattr(model, "set_mode"):
        model.set_mode(mode)

    embeddings = []
    batch_size = 8

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(_embed_device)

        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"]
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            batch_emb = (sum_hidden / sum_mask).cpu().float().tolist()
            embeddings.extend(batch_emb)

    return embeddings

# --- Local Reranking Model ---

_reranker = None
_reranker_tokenizer = None

_reranker_device = None

def get_reranker():
    global _reranker, _reranker_tokenizer, _reranker_device
    if _reranker is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        out(f"Loading reranker: {RERANK_MODEL}")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, trust_remote_code=True)
        _reranker = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, trust_remote_code=True)
        _reranker_device = get_device()
        _reranker = _reranker.to(_reranker_device).half()
        out(f"Reranker loaded on {_reranker_device}")
    return _reranker, _reranker_tokenizer

def rerank(query: str, docs: List[Dict], top_k: int = 10) -> List[Dict]:
    import torch
    if not docs:
        return docs[:top_k]

    model, tokenizer = get_reranker()

    pairs = []
    for d in docs:
        text = d.get("text", d.get("title", ""))[:2000]
        pairs.append([query, text])

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(_reranker_device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

# --- Zotero API ---

def zotero_api(path, params=None):
    import requests
    r = requests.get(f"http://localhost:23119/api/users/0/{path}", params=params or {})
    r.raise_for_status()
    return r.json(), int(r.headers.get("Last-Modified-Version", 0))

def get_collections():
    items, _ = zotero_api("collections")
    return {c["data"]["name"]: c["data"]["key"] for c in items}

def get_collection_items(collection_key):
    items, _ = zotero_api(f"collections/{collection_key}/items", {"itemType": "-attachment"})
    return [it["data"]["key"] for it in items]

def get_pdf_items(limit=None, keys=None):
    params = {"itemType": "-attachment"}
    if limit:
        params["limit"] = limit
    items, ver = zotero_api("items", params)

    result = []
    for it in items:
        key = it["data"]["key"]
        if keys and key not in keys:
            continue
        title = it["data"].get("title", "")
        authors = ", ".join([c.get("lastName", "") for c in it["data"].get("creators", [])[:3]])
        year = (it["data"].get("date") or "")[:4]
        abstract = it["data"].get("abstractNote", "")
        children, _ = zotero_api(f"items/{key}/children")
        for c in children:
            if c["data"].get("contentType") == "application/pdf":
                link = c.get("links", {}).get("enclosure", {})
                pdf_path = urllib.parse.unquote(link.get("href", "").replace("file://", ""))
                if pdf_path and os.path.exists(pdf_path):
                    result.append({
                        "key": key, "title": title, "authors": authors,
                        "year": int(year) if year.isdigit() else 0,
                        "abstract": abstract, "pdf": pdf_path
                    })
                    break
    return result, ver

def extract_text(pdf_path, max_pages=10):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text_parts = []
        for i, page in enumerate(doc):
            if i >= max_pages: break
            text_parts.append(page.get_text())
        doc.close()
        text = "\n".join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:30000]
    except:
        return ""

# --- LanceDB ---

def get_db():
    import lancedb
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DATA_DIR))

# --- Commands ---

def cmd_collections():
    collections = get_collections()
    print(f"\n📚 Zotero Collections ({len(collections)}):\n")
    for name, key in sorted(collections.items()):
        print(f"  • {name} [{key}]")

def cmd_build(limit=None, collection=None):
    from tqdm import tqdm

    keys = None
    if collection:
        collections = get_collections()
        if collection not in collections:
            out(f"Collection '{collection}' not found.")
            return
        keys = set(get_collection_items(collections[collection]))
        out(f"Building index for collection: {collection} ({len(keys)} items)")

    out("Fetching papers from Zotero...")
    items, ver = get_pdf_items(limit=limit, keys=keys)
    out(f"Found {len(items)} papers with PDFs")
    if not items: return

    db = get_db()
    try:
        db.drop_table(TABLE_NAME)
    except:
        pass

    records = []
    for item in tqdm(items, desc="Extracting text"):
        content = extract_text(item["pdf"], max_pages=10)
        text = f"{item['title']} {item['authors']} {item['abstract']} {content}"
        if text.strip():
            records.append({
                "key": item["key"],
                "title": item["title"],
                "authors": item["authors"],
                "year": item["year"],
                "text": text[:30000],
            })

    out(f"Embedding {len(records)} papers...")
    texts = [r["text"][:8000] for r in records]
    embeddings = []

    for i in tqdm(range(0, len(texts), 8), desc="Embedding"):
        batch = texts[i:i+8]
        try:
            embeddings.extend(get_embeddings(batch, mode="document"))
        except Exception as e:
            out(f"Error: {e}")
            for t in batch:
                try:
                    embeddings.extend(get_embeddings([t], mode="document"))
                except:
                    embeddings.append([0.0] * _embed_dim if _embed_dim else [0.0] * 1024)

    for r, emb in zip(records, embeddings):
        r["vector"] = emb

    db.create_table(TABLE_NAME, records)

    meta = {"version": ver, "count": len(records), "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL}
    if collection:
        meta["collection"] = collection
    (DATA_DIR / "meta.json").write_text(json.dumps(meta))

    out(f"Done! Indexed {len(records)} papers")

def cmd_sync():
    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)
    existing = set(r["key"] for r in table.to_pandas().to_dict("records"))

    items, ver = get_pdf_items()
    new_items = [it for it in items if it["key"] not in existing]

    if not new_items:
        out("Library up to date.")
        return

    out(f"Found {len(new_items)} new papers")
    from tqdm import tqdm

    records = []
    for item in tqdm(new_items, desc="Processing"):
        content = extract_text(item["pdf"], max_pages=10)
        text = f"{item['title']} {item['authors']} {item['abstract']} {content}"
        if text.strip():
            records.append({
                "key": item["key"],
                "title": item["title"],
                "authors": item["authors"],
                "year": item["year"],
                "text": text[:30000],
            })

    if records:
        texts = [r["text"][:8000] for r in records]
        embeddings = get_embeddings(texts, mode="document")
        for r, emb in zip(records, embeddings):
            r["vector"] = emb
        table.add(records)

    meta = json.loads((DATA_DIR / "meta.json").read_text())
    meta["version"] = ver
    meta["count"] = meta.get("count", 0) + len(records)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta))

    out(f"Added {len(records)} papers")

def cmd_search(query: str, top_k: int = 10, year_min: Optional[int] = None, keys: set = None, add_to: str = None):
    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)

    out(f"Searching: {query}")
    start = time.time()

    query_emb = get_embeddings([query], mode="query")[0]
    results = table.search(query_emb).limit(top_k * 3 if keys else top_k * 2)

    if year_min:
        results = results.where(f"year >= {year_min}")

    candidates = results.to_pandas().to_dict("records")

    if keys:
        candidates = [c for c in candidates if c["key"] in keys]

    elapsed = time.time() - start

    output = []
    result_keys = []
    for r in candidates[:top_k]:
        result_keys.append(r["key"])
        output.append({
            "key": r["key"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
            "score": round(float(r.get("_distance", 0)), 4),
        })

    print(json.dumps({"query": query, "results": output, "time_ms": int(elapsed*1000)}, ensure_ascii=False, indent=2))

    if add_to and result_keys:
        ws = load_workspaces()
        if add_to not in ws:
            ws[add_to] = {"keys": [], "created": time.strftime("%Y-%m-%d %H:%M")}
            out(f"Created workspace: {add_to}")

        existing = set(ws[add_to]["keys"])
        added = [k for k in result_keys if k not in existing]
        ws[add_to]["keys"].extend(added)
        save_workspaces(ws)
        out(f"Added {len(added)} papers to workspace '{add_to}'")

# --- Workspace Commands ---

def cmd_ws_list():
    ws = load_workspaces()
    if not ws:
        print("\nNo workspaces. Create one with: workspace.py ws-create <name>")
        return

    print(f"\n📂 Workspaces ({len(ws)}):\n")
    for name, data in ws.items():
        count = len(data.get("keys", []))
        print(f"  • {name} ({count} papers)")

def cmd_ws_create(name):
    ws = load_workspaces()
    if name in ws:
        out(f"Workspace '{name}' already exists.")
        return

    ws[name] = {"keys": [], "created": time.strftime("%Y-%m-%d %H:%M")}
    save_workspaces(ws)
    out(f"Created workspace: {name}")

def cmd_ws_add(name, keys_str):
    ws = load_workspaces()
    if name not in ws:
        out(f"Workspace '{name}' not found. Create it first.")
        return

    keys = [k.strip() for k in keys_str.split(",")]
    existing = set(ws[name]["keys"])
    added = [k for k in keys if k not in existing]
    ws[name]["keys"].extend(added)
    save_workspaces(ws)
    out(f"Added {len(added)} papers to '{name}' (total: {len(ws[name]['keys'])})")

def cmd_ws_import(name, collection_name):
    ws = load_workspaces()
    if name not in ws:
        ws[name] = {"keys": [], "created": time.strftime("%Y-%m-%d %H:%M")}

    collections = get_collections()
    if collection_name not in collections:
        out(f"Collection '{collection_name}' not found.")
        cmd_collections()
        return

    keys = get_collection_items(collections[collection_name])
    existing = set(ws[name]["keys"])
    added = [k for k in keys if k not in existing]
    ws[name]["keys"].extend(added)
    save_workspaces(ws)
    out(f"Imported {len(added)} papers from '{collection_name}' to workspace '{name}'")

def cmd_ws_search(name, query, top_k=10):
    ws = load_workspaces()
    if name not in ws:
        out(f"Workspace '{name}' not found.")
        return

    keys = set(ws[name]["keys"])
    if not keys:
        out(f"Workspace '{name}' is empty.")
        return

    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)

    out(f"Searching workspace '{name}' ({len(keys)} papers)")
    start = time.time()

    query_emb = get_embeddings([query], mode="query")[0]
    candidates = table.search(query_emb).limit(len(keys) * 2).to_pandas().to_dict("records")
    candidates = [c for c in candidates if c["key"] in keys]

    if len(candidates) > 0:
        out(f"Reranking with {RERANK_MODEL}...")
        candidates = rerank(query, candidates, top_k=top_k)

    elapsed = time.time() - start

    output = []
    for r in candidates:
        output.append({
            "key": r["key"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
            "rerank_score": round(float(r.get("rerank_score", 0)), 4),
        })

    print(json.dumps({
        "query": query,
        "workspace": name,
        "reranked": True,
        "results": output,
        "time_ms": int(elapsed*1000)
    }, ensure_ascii=False, indent=2))

def cmd_ws_delete(name):
    ws = load_workspaces()
    if name not in ws:
        out(f"Workspace '{name}' not found.")
        return

    del ws[name]
    save_workspaces(ws)
    out(f"Deleted workspace: {name}")

def cmd_status():
    print("\n=== Zotero Research Assistant ===\n")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Reranker:  {RERANK_MODEL}")

    if (DATA_DIR / "meta.json").exists():
        meta = json.loads((DATA_DIR / "meta.json").read_text())
        size_mb = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"\nIndex: {meta.get('count', 0)} papers ({size_mb:.1f} MB)")
        if "collection" in meta:
            print(f"Collection: {meta['collection']}")
    else:
        print("\nNo index. Run 'build' first.")

    ws = load_workspaces()
    if ws:
        print(f"\nWorkspaces: {len(ws)}")
        for name, data in ws.items():
            print(f"  • {name} ({len(data.get('keys', []))} papers)")

def cmd_delete():
    import shutil
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        out("Deleted all data")
    else:
        out("No data found")

def main():
    parser = argparse.ArgumentParser(description="Zotero + LanceDB + Qwen3 Models")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("collections", help="List Zotero collections")

    p_build = sub.add_parser("build", help="Build index")
    p_build.add_argument("--limit", type=int, help="Max papers")
    p_build.add_argument("--collection", type=str, help="Only index this collection")

    sub.add_parser("sync", help="Sync new papers")

    p_search = sub.add_parser("search", help="Search all papers")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=10, help="Top K")
    p_search.add_argument("--year", type=int, help="Min year")
    p_search.add_argument("--add-to", type=str, help="Add results to workspace")

    sub.add_parser("ws-list", help="List workspaces")

    p_ws_create = sub.add_parser("ws-create", help="Create workspace")
    p_ws_create.add_argument("name", help="Workspace name")

    p_ws_add = sub.add_parser("ws-add", help="Add papers to workspace")
    p_ws_add.add_argument("name", help="Workspace name")
    p_ws_add.add_argument("keys", help="Comma-separated paper keys")

    p_ws_import = sub.add_parser("ws-import", help="Import collection to workspace")
    p_ws_import.add_argument("name", help="Workspace name")
    p_ws_import.add_argument("collection", help="Collection name")

    p_ws_search = sub.add_parser("ws-search", help="Search within workspace")
    p_ws_search.add_argument("name", help="Workspace name")
    p_ws_search.add_argument("query", help="Search query")
    p_ws_search.add_argument("-k", type=int, default=10, help="Top K")

    p_ws_delete = sub.add_parser("ws-delete", help="Delete workspace")
    p_ws_delete.add_argument("name", help="Workspace name")

    sub.add_parser("status", help="Show status")
    sub.add_parser("delete", help="Delete all data")

    args = parser.parse_args()

    if args.cmd == "collections":
        cmd_collections()
    elif args.cmd == "build":
        cmd_build(args.limit, args.collection)
    elif args.cmd == "sync":
        cmd_sync()
    elif args.cmd == "search":
        cmd_search(args.query, args.k, args.year, add_to=args.add_to)
    elif args.cmd == "ws-list":
        cmd_ws_list()
    elif args.cmd == "ws-create":
        cmd_ws_create(args.name)
    elif args.cmd == "ws-add":
        cmd_ws_add(args.name, args.keys)
    elif args.cmd == "ws-import":
        cmd_ws_import(args.name, args.collection)
    elif args.cmd == "ws-search":
        cmd_ws_search(args.name, args.query, args.k)
    elif args.cmd == "ws-delete":
        cmd_ws_delete(args.name)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "delete":
        cmd_delete()
    else:
        parser.print_help()
        print("\nStart with: workspace.py build")

if __name__ == "__main__":
    main()
