#!/usr/bin/env python3
"""Zotero Research Assistant - LanceDB + Local Qwen3 Models.

Usage:
  python workspace.py build [--collection NAME]  Build index (all or collection)
  python workspace.py sync                       Sync new papers
  python workspace.py search <query> [-k] [--add-to WS]  Search (optionally add to workspace)

  # Browse commands
  python workspace.py list [--limit N]           List papers in index
  python workspace.py get <key>                  Get paper details

  # Workspace commands
  python workspace.py ws-list                    List workspaces
  python workspace.py ws-create <name>           Create workspace
  python workspace.py ws-add <name> <keys>       Add papers to workspace
  python workspace.py ws-remove <name> <keys>    Remove papers from workspace
  python workspace.py ws-import <name> <coll>    Import collection to workspace
  python workspace.py ws-search <name> <query>   Search within workspace
  python workspace.py ws-show <name>             Show papers in workspace
  python workspace.py ws-delete <name>           Delete workspace

  # Interactive
  python workspace.py shell                      Interactive mode

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

# Model API configuration
MODEL_API = "http://127.0.0.1:8765"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = None  # Fetched from API

def out(msg): print(f"[zotero] {msg}", flush=True)

# --- Table Formatting ---

def format_table(headers, rows, max_width=50):
    """Format data as a simple ASCII table."""
    def truncate(s, w):
        s = str(s)
        return s[:w-2] + ".." if len(s) > w else s

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = min(max(widths[i], len(str(cell))), max_width)

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    lines = [fmt.format(*[truncate(h, widths[i]) for i, h in enumerate(headers)]), sep]
    for row in rows:
        lines.append(fmt.format(*[truncate(cell, widths[i]) for i, cell in enumerate(row)]))
    return "\n".join(lines)

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

# --- Embedding & Reranking API ---

def check_model_api():
    import requests
    try:
        r = requests.get(f"{MODEL_API}/health", timeout=2)
        return r.ok
    except:
        return False

def get_reranker_info():
    import requests
    try:
        r = requests.get(f"{MODEL_API}/health", timeout=2)
        if r.ok:
            return r.json().get("reranker", "unknown")
    except:
        pass
    return "unknown"

def get_embeddings(texts: List[str], mode: str = "document") -> List[List[float]]:
    import requests
    try:
        r = requests.post(f"{MODEL_API}/embed", json={"texts": texts, "mode": mode}, timeout=120)
        r.raise_for_status()
        return r.json()["embeddings"]
    except requests.exceptions.ConnectionError:
        out(f"Error: Model API not running. Start it with:")
        out(f"  python model_server.py --preload --port 8765")
        sys.exit(1)

def rerank(query: str, docs: List[Dict], top_k: int = 10) -> List[Dict]:
    import requests
    if not docs:
        return docs[:top_k]

    doc_texts = [d.get("text", d.get("title", ""))[:2000] for d in docs]

    try:
        r = requests.post(f"{MODEL_API}/rerank", json={"query": query, "docs": doc_texts, "top_k": top_k}, timeout=60)
        r.raise_for_status()
        results = r.json()["results"]
    except requests.exceptions.ConnectionError:
        out(f"Error: Model API not running. Start it with:")
        out(f"  python model_server.py --preload --port 8765")
        sys.exit(1)

    for res in results:
        docs[res["index"]]["rerank_score"] = res["score"]

    reranked = sorted(docs, key=lambda x: x.get("rerank_score", -999), reverse=True)
    return reranked[:top_k]

# --- Zotero API ---

ZOTERO_URL = "http://localhost:23119"

def check_zotero():
    import requests
    try:
        r = requests.get(f"{ZOTERO_URL}/api/users/0/items", params={"limit": 1}, timeout=3)
        return r.ok
    except:
        return False

def zotero_api(path, params=None):
    import requests
    try:
        r = requests.get(f"{ZOTERO_URL}/api/users/0/{path}", params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json(), int(r.headers.get("Last-Modified-Version", 0))
    except requests.exceptions.ConnectionError:
        out("Error: Cannot connect to Zotero. Is it running?")
        out("Start Zotero and ensure it's listening on localhost:23119")
        sys.exit(1)
    except requests.exceptions.Timeout:
        out("Error: Zotero API timeout")
        sys.exit(1)

def zotero_api_all(path, params=None, page_size=100):
    """Fetch all items with pagination."""
    params = dict(params or {})
    params["limit"] = page_size
    all_items, start, ver = [], 0, 0
    while True:
        params["start"] = start
        items, ver = zotero_api(path, params)
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_size:
            break
        start += page_size
    return all_items, ver

def get_collections():
    items, _ = zotero_api_all("collections")
    return {c["data"]["name"]: c["data"]["key"] for c in items}

def get_collection_items(collection_key):
    items, _ = zotero_api_all(f"collections/{collection_key}/items", {"itemType": "-attachment"})
    return [it["data"]["key"] for it in items]

def get_pdf_items(limit=None, keys=None):
    params = {"itemType": "-attachment"}
    if limit:
        items, ver = zotero_api("items", {**params, "limit": limit})
    else:
        items, ver = zotero_api_all("items", params)

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
    if not check_model_api():
        out("Error: Model API not running. Start it with:")
        out("  python model_server.py --preload --port 8765")
        return

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
                "pdf_path": item["pdf"],
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
                    embeddings.append([0.0] * 1024)

    for r, emb in zip(records, embeddings):
        r["vector"] = emb

    db.create_table(TABLE_NAME, records)

    meta = {"version": ver, "count": len(records), "embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL}
    if collection:
        meta["collection"] = collection
    (DATA_DIR / "meta.json").write_text(json.dumps(meta))

    out(f"Done! Indexed {len(records)} papers")

def cmd_sync():
    if not check_model_api():
        out("Error: Model API not running. Start it with:")
        out("  python model_server.py --preload --port 8765")
        return

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
                "pdf_path": item["pdf"],
                "text": text[:30000],
            })

    if records:
        out(f"Embedding {len(records)} papers...")
        texts = [r["text"][:8000] for r in records]
        embeddings = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            try:
                embeddings.extend(get_embeddings(batch, mode="document"))
            except Exception as e:
                out(f"Error: {e}")
                for t in batch:
                    try:
                        embeddings.extend(get_embeddings([t], mode="document"))
                    except:
                        embeddings.append([0.0] * 1024)
        for r, emb in zip(records, embeddings):
            r["vector"] = emb
        table.add(records)

    meta = json.loads((DATA_DIR / "meta.json").read_text())
    meta["version"] = ver
    meta["count"] = meta.get("count", 0) + len(records)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta))

    out(f"Added {len(records)} papers")

def cmd_search(query: str, top_k: int = 10, year_min: Optional[int] = None, keys: set = None, add_to: str = None, use_rerank: bool = False):
    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)

    out(f"Searching: {query}")
    start = time.time()

    query_emb = get_embeddings([query], mode="query")[0]
    fetch_limit = top_k * 3 if (keys or use_rerank) else top_k * 2
    results = table.search(query_emb).limit(fetch_limit)

    if year_min:
        results = results.where(f"year >= {year_min}")

    candidates = results.to_pandas().to_dict("records")

    if keys:
        candidates = [c for c in candidates if c["key"] in keys]

    if use_rerank and len(candidates) > 1:
        out(f"Reranking with {get_reranker_info()}...")
        candidates = rerank(query, candidates, top_k=top_k)

    elapsed = time.time() - start

    output = []
    result_keys = []
    for r in candidates[:top_k]:
        result_keys.append(r["key"])
        item = {
            "key": r["key"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
        }
        if use_rerank:
            item["rerank_score"] = round(float(r.get("rerank_score", 0)), 4)
        else:
            dist = float(r.get("_distance", 0))
            item["similarity"] = round(1.0 - dist, 4)
        output.append(item)

    print(json.dumps({"query": query, "reranked": use_rerank, "results": output, "time_ms": int(elapsed*1000)}, ensure_ascii=False, indent=2))

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

def cmd_list(limit=20, offset=0, sort_by="year"):
    """List papers in the index."""
    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    if sort_by == "year":
        df = df.sort_values("year", ascending=False)
    elif sort_by == "title":
        df = df.sort_values("title")

    total = len(df)
    df = df.iloc[offset:offset+limit]
    papers = df.to_dict("records")

    print(f"\n📚 Papers in Index ({offset+1}-{min(offset+limit, total)} of {total})\n")

    headers = ["Key", "Year", "Authors", "Title"]
    rows = []
    for p in papers:
        rows.append([
            p["key"],
            p.get("year", ""),
            p.get("authors", "")[:25],
            p.get("title", "")[:50]
        ])

    print(format_table(headers, rows))

    if offset + limit < total:
        print(f"\n  → Next page: list --offset {offset+limit}")

def cmd_get(key: str):
    """Get full metadata for a paper (from Zotero API + index)."""
    import requests
    # Try Zotero API first for rich metadata
    try:
        r = requests.get(f"{ZOTERO_URL}/api/users/0/items/{key}", timeout=10)
        r.raise_for_status()
        d = r.json()["data"]
        children_r = requests.get(f"{ZOTERO_URL}/api/users/0/items/{key}/children", timeout=10)
        children = children_r.json() if children_r.ok else []
        pdfs = []
        for c in children:
            if c["data"].get("contentType") == "application/pdf":
                link = c.get("links", {}).get("enclosure", {})
                path = urllib.parse.unquote(link.get("href", "").replace("file://", ""))
                pdfs.append({"key": c["data"]["key"], "path": path})

        creators = ", ".join(c.get("lastName", c.get("name", "")) for c in d.get("creators", []))
        tags = [t["tag"] for t in d.get("tags", [])]
        result = {
            "key": d["key"], "type": d["itemType"], "title": d.get("title", ""),
            "creators": creators, "date": d.get("date", ""),
            "abstract": d.get("abstractNote", ""), "tags": tags,
            "DOI": d.get("DOI", ""), "url": d.get("url", ""),
            "publication": d.get("publicationTitle", ""),
            "pdfs": pdfs,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except:
        # Fallback to index
        if not (DATA_DIR / "meta.json").exists():
            out(f"Paper '{key}' not found.")
            return
        db = get_db()
        table = db.open_table(TABLE_NAME)
        df = table.to_pandas()
        matches = df[df["key"] == key].to_dict("records")
        if not matches:
            out(f"Paper '{key}' not found in index.")
            return
        paper = matches[0]
        print(json.dumps({
            "key": paper["key"], "title": paper["title"],
            "authors": paper["authors"], "year": paper["year"],
        }, ensure_ascii=False, indent=2))

def cmd_tags(limit=100):
    """List all tags in Zotero library."""
    items, _ = zotero_api("tags", {"limit": limit})
    tags = [t["tag"] for t in items]
    print(json.dumps(tags, ensure_ascii=False, indent=2))

def cmd_fulltext(key: str):
    """Get Zotero's indexed full-text content for a paper."""
    children, _ = zotero_api(f"items/{key}/children")
    for c in children:
        if c["data"].get("contentType") == "application/pdf":
            try:
                ft, _ = zotero_api(f"items/{c['data']['key']}/fulltext")
                content = ft.get("content", "")
                if content:
                    print(content[:5000])
                    return
            except:
                pass
    out(f"No full-text content found for '{key}'")

def cmd_meta_search(query, limit=20):
    """Search Zotero metadata (title/creator/tag) without vector index."""
    items, _ = zotero_api("items", {"q": query, "limit": limit, "itemType": "-attachment"})
    results = []
    for it in items:
        d = it["data"]
        creators = ", ".join(c.get("lastName", c.get("name", "")) for c in d.get("creators", []))
        results.append({
            "key": d["key"], "type": d["itemType"], "title": d.get("title", ""),
            "creators": creators, "date": d.get("date", ""),
            "tags": [t["tag"] for t in d.get("tags", [])],
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))

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

def cmd_ws_search(name, query, top_k=10, use_rerank=False):
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

    if use_rerank and len(candidates) > 1:
        out(f"Reranking with {get_reranker_info()}...")
        candidates = rerank(query, candidates, top_k=top_k)

    elapsed = time.time() - start

    output = []
    for r in candidates[:top_k]:
        item = {
            "key": r["key"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
        }
        if use_rerank:
            item["rerank_score"] = round(float(r.get("rerank_score", 0)), 4)
        else:
            dist = float(r.get("_distance", 0))
            item["similarity"] = round(1.0 - dist, 4)
        output.append(item)

    print(json.dumps({
        "query": query,
        "workspace": name,
        "reranked": use_rerank,
        "results": output,
        "time_ms": int(elapsed*1000)
    }, ensure_ascii=False, indent=2))

def cmd_ws_remove(name, keys_str):
    """Remove papers from a workspace."""
    ws = load_workspaces()
    if name not in ws:
        out(f"Workspace '{name}' not found.")
        return

    keys_to_remove = set(k.strip() for k in keys_str.split(","))
    before = len(ws[name]["keys"])
    ws[name]["keys"] = [k for k in ws[name]["keys"] if k not in keys_to_remove]
    removed = before - len(ws[name]["keys"])
    save_workspaces(ws)
    out(f"Removed {removed} papers from '{name}' (remaining: {len(ws[name]['keys'])})")

def cmd_ws_show(name):
    """Show all papers in a workspace with details."""
    ws = load_workspaces()
    if name not in ws:
        out(f"Workspace '{name}' not found.")
        return

    keys = ws[name]["keys"]
    if not keys:
        print(f"\nWorkspace '{name}' is empty.")
        return

    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()

    key_set = set(keys)
    papers = df[df["key"].isin(key_set)].to_dict("records")

    print(f"\n📂 Workspace: {name} ({len(papers)} papers)\n")

    headers = ["Key", "Year", "Authors", "Title"]
    rows = []
    for p in sorted(papers, key=lambda x: x.get("year", 0), reverse=True):
        rows.append([
            p["key"],
            p.get("year", ""),
            p.get("authors", "")[:25],
            p.get("title", "")[:50]
        ])

    print(format_table(headers, rows))

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
    print(f"Reranker:  {get_reranker_info()}")

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

def cmd_delete(force=False):
    import shutil
    if not DATA_DIR.exists():
        out("No data found")
        return

    if not force:
        meta = json.loads((DATA_DIR / "meta.json").read_text()) if (DATA_DIR / "meta.json").exists() else {}
        count = meta.get("count", 0)
        ws_count = len(load_workspaces())
        print(f"\n⚠️  This will delete:")
        print(f"   • {count} indexed papers")
        print(f"   • {ws_count} workspaces")
        print(f"   • All embeddings and metadata\n")
        try:
            confirm = input("Type 'yes' to confirm: ").strip()
            if confirm.lower() != "yes":
                out("Cancelled")
                return
        except KeyboardInterrupt:
            print("\nCancelled")
            return

    shutil.rmtree(DATA_DIR)
    out("Deleted all data")

# --- Interactive Shell ---

def cmd_shell():
    """Interactive shell mode."""
    try:
        import readline
        readline.parse_and_bind("tab: complete")
    except ImportError:
        pass

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║       Zotero Research Assistant - Interactive        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Commands:                                           ║")
    print("║    search <query>      - Search papers               ║")
    print("║    list                - List papers                 ║")
    print("║    get <key>           - Get paper details           ║")
    print("║    ws-list             - List workspaces             ║")
    print("║    ws-show <name>      - Show workspace papers       ║")
    print("║    ws-search <name> q  - Search in workspace         ║")
    print("║    status              - Show status                 ║")
    print("║    help                - Show this help              ║")
    print("║    exit/quit           - Exit shell                  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    history = []

    while True:
        try:
            line = input("zotero> ").strip()
            if not line:
                continue

            history.append(line)
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            elif cmd == "help":
                print("Commands: search, list, get, ws-list, ws-show, ws-search, status, exit")
            elif cmd == "search":
                if arg:
                    cmd_search(arg, top_k=10, use_rerank=True)
                else:
                    print("Usage: search <query>")
            elif cmd == "list":
                cmd_list()
            elif cmd == "get":
                if arg:
                    cmd_get(arg)
                else:
                    print("Usage: get <key>")
            elif cmd == "ws-list":
                cmd_ws_list()
            elif cmd == "ws-show":
                if arg:
                    cmd_ws_show(arg)
                else:
                    print("Usage: ws-show <workspace>")
            elif cmd == "ws-search":
                ws_parts = arg.split(maxsplit=1)
                if len(ws_parts) == 2:
                    cmd_ws_search(ws_parts[0], ws_parts[1], use_rerank=True)
                else:
                    print("Usage: ws-search <workspace> <query>")
            elif cmd == "status":
                cmd_status()
            elif cmd == "history":
                for i, h in enumerate(history[-10:], 1):
                    print(f"  {i}. {h}")
            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")

        except KeyboardInterrupt:
            print("\n(Ctrl+C to exit, type 'exit' to quit)")
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

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
    p_search.add_argument("--rerank", action="store_true", help="Use neural reranking")

    p_list = sub.add_parser("list", help="List papers in index")
    p_list.add_argument("--limit", type=int, default=20, help="Max papers to show")
    p_list.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    p_list.add_argument("--sort", choices=["year", "title"], default="year", help="Sort by")

    p_get = sub.add_parser("get", help="Get paper details")
    p_get.add_argument("key", help="Paper key")

    p_meta = sub.add_parser("meta-search", help="Search Zotero metadata (no vector index needed)")
    p_meta.add_argument("query", help="Search query")
    p_meta.add_argument("--limit", type=int, default=20, help="Max results")

    p_tags = sub.add_parser("tags", help="List all Zotero tags")
    p_tags.add_argument("--limit", type=int, default=100, help="Max tags")

    p_fulltext = sub.add_parser("fulltext", help="Get Zotero full-text for a paper")
    p_fulltext.add_argument("key", help="Paper key")

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
    p_ws_search.add_argument("--rerank", action="store_true", help="Use neural reranking")

    p_ws_remove = sub.add_parser("ws-remove", help="Remove papers from workspace")
    p_ws_remove.add_argument("name", help="Workspace name")
    p_ws_remove.add_argument("keys", help="Comma-separated paper keys")

    p_ws_show = sub.add_parser("ws-show", help="Show papers in workspace")
    p_ws_show.add_argument("name", help="Workspace name")

    p_ws_delete = sub.add_parser("ws-delete", help="Delete workspace")
    p_ws_delete.add_argument("name", help="Workspace name")

    sub.add_parser("status", help="Show status")
    sub.add_parser("shell", help="Interactive mode")

    p_delete = sub.add_parser("delete", help="Delete all data")
    p_delete.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    if args.cmd == "collections":
        cmd_collections()
    elif args.cmd == "build":
        cmd_build(args.limit, args.collection)
    elif args.cmd == "sync":
        cmd_sync()
    elif args.cmd == "search":
        cmd_search(args.query, args.k, args.year, add_to=args.add_to, use_rerank=args.rerank)
    elif args.cmd == "list":
        cmd_list(args.limit, args.offset, args.sort)
    elif args.cmd == "get":
        cmd_get(args.key)
    elif args.cmd == "meta-search":
        cmd_meta_search(args.query, args.limit)
    elif args.cmd == "tags":
        cmd_tags(args.limit)
    elif args.cmd == "fulltext":
        cmd_fulltext(args.key)
    elif args.cmd == "ws-list":
        cmd_ws_list()
    elif args.cmd == "ws-create":
        cmd_ws_create(args.name)
    elif args.cmd == "ws-add":
        cmd_ws_add(args.name, args.keys)
    elif args.cmd == "ws-import":
        cmd_ws_import(args.name, args.collection)
    elif args.cmd == "ws-search":
        cmd_ws_search(args.name, args.query, args.k, use_rerank=args.rerank)
    elif args.cmd == "ws-remove":
        cmd_ws_remove(args.name, args.keys)
    elif args.cmd == "ws-show":
        cmd_ws_show(args.name)
    elif args.cmd == "ws-delete":
        cmd_ws_delete(args.name)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "shell":
        cmd_shell()
    elif args.cmd == "delete":
        cmd_delete(args.force)
    else:
        parser.print_help()
        print("\nStart with: workspace.py build")

if __name__ == "__main__":
    main()
