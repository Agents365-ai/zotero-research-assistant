#!/usr/bin/env python3
"""Zotero Research Assistant - LanceDB + Qwen3 Embedding/Reranker.

Usage:
  python workspace.py build [--limit N]   Build index from Zotero PDFs
  python workspace.py sync                Sync new papers
  python workspace.py search <query> [-k] Search with reranking
  python workspace.py status              Show index status
  python workspace.py delete              Delete index
"""
import sys, os, json, argparse, re, time
from pathlib import Path
from typing import Optional
import urllib.parse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = Path.home() / ".local/share/zotero-lance"
TABLE_NAME = "papers"

def out(msg): print(f"[zotero] {msg}", flush=True)

def get_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

# --- Zotero API ---

def zotero_api(path, params=None):
    import requests
    r = requests.get(f"http://localhost:23119/api/users/0/{path}", params=params or {})
    r.raise_for_status()
    return r.json(), int(r.headers.get("Last-Modified-Version", 0))

def get_pdf_items(limit=500):
    items, ver = zotero_api("items", {"limit": limit, "itemType": "-attachment"})
    result = []
    for it in items:
        key = it["data"]["key"]
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

# --- Qwen3 Models ---

_embed_model = None
_rerank_model = None

def load_embed_model():
    global _embed_model
    if _embed_model is not None: return _embed_model
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "Qwen/Qwen3-Embedding-4B"
    out(f"Loading {model_name} on {get_device()}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(get_device())
    model.requires_grad_(False)

    out(f"Embedding model loaded in {time.time()-start:.1f}s")
    _embed_model = (model, tokenizer)
    return _embed_model

def load_rerank_model():
    global _rerank_model
    if _rerank_model is not None: return _rerank_model
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "Qwen/Qwen3-Reranker-4B"
    out(f"Loading {model_name} on {get_device()}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(get_device())
    model.requires_grad_(False)

    out(f"Reranker model loaded in {time.time()-start:.1f}s")
    _rerank_model = (model, tokenizer)
    return _rerank_model

def embed_texts(texts, batch_size=4):
    import torch
    model, tokenizer = load_embed_model()
    device = get_device()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            all_embeddings.extend(mean_embeddings.cpu().numpy().tolist())

    return all_embeddings

def rerank(query, docs, top_k=10):
    import torch
    model, tokenizer = load_rerank_model()
    device = get_device()

    pairs = [[query, doc["text"]] for doc in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=4096, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1).cpu().numpy().tolist()

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = score

    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# --- LanceDB ---

def get_db():
    import lancedb
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(DATA_DIR))

def cmd_build(limit=None):
    import pyarrow as pa
    from tqdm import tqdm

    out("Fetching papers from Zotero...")
    items, ver = get_pdf_items(limit=limit or 500)
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

    out(f"Extracted {len(records)} papers, now embedding...")

    texts = [r["text"] for r in records]
    embeddings = []
    batch_size = 4

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        batch_emb = embed_texts(batch, batch_size=batch_size)
        embeddings.extend(batch_emb)

    for r, emb in zip(records, embeddings):
        r["vector"] = emb

    table = db.create_table(TABLE_NAME, records)

    meta = {"version": ver, "count": len(records)}
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
        texts = [r["text"] for r in records]
        embeddings = embed_texts(texts)
        for r, emb in zip(records, embeddings):
            r["vector"] = emb
        table.add(records)

    meta = json.loads((DATA_DIR / "meta.json").read_text())
    meta["version"] = ver
    meta["count"] = meta.get("count", 0) + len(records)
    (DATA_DIR / "meta.json").write_text(json.dumps(meta))

    out(f"Added {len(records)} papers")

def cmd_search(query: str, top_k: int = 10, year_min: Optional[int] = None):
    if not (DATA_DIR / "meta.json").exists():
        out("No index found. Run 'build' first.")
        return

    db = get_db()
    table = db.open_table(TABLE_NAME)

    out(f"Searching: {query}")
    start = time.time()

    query_emb = embed_texts([query])[0]

    results = table.search(query_emb).limit(100)

    if year_min:
        results = results.where(f"year >= {year_min}")

    candidates = results.to_pandas().to_dict("records")

    if not candidates:
        print(json.dumps({"query": query, "results": []}, ensure_ascii=False, indent=2))
        return

    out(f"Reranking {len(candidates)} candidates...")
    reranked = rerank(query, candidates, top_k=top_k)

    elapsed = time.time() - start

    output = []
    for r in reranked:
        output.append({
            "key": r["key"],
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year"],
            "score": round(r["rerank_score"], 3),
        })

    print(json.dumps({"query": query, "results": output, "time_ms": int(elapsed*1000)}, ensure_ascii=False, indent=2))

def cmd_status():
    if not (DATA_DIR / "meta.json").exists():
        out("No index found.")
        return

    meta = json.loads((DATA_DIR / "meta.json").read_text())
    db = get_db()
    table = db.open_table(TABLE_NAME)

    size_mb = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()) / 1024 / 1024

    print(f"""
=== Zotero-LanceDB Index Status ===
Papers indexed: {meta.get('count', 0)}
Library version: {meta.get('version', 'unknown')}
Database size: {size_mb:.1f} MB
Embedding model: Qwen3-Embedding-4B
Reranker model: Qwen3-Reranker-4B
Data path: {DATA_DIR}
""")

def cmd_delete():
    import shutil
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        out("Deleted database")
    else:
        out("No index found")

def main():
    parser = argparse.ArgumentParser(description="Zotero + LanceDB + Qwen3")
    sub = parser.add_subparsers(dest="cmd")

    p_build = sub.add_parser("build", help="Build index")
    p_build.add_argument("--limit", type=int, help="Max papers to index")

    sub.add_parser("sync", help="Sync new papers")

    p_search = sub.add_parser("search", help="Search papers")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=10, help="Top K results")
    p_search.add_argument("--year", type=int, help="Min year filter")

    sub.add_parser("status", help="Show index status")
    sub.add_parser("delete", help="Delete index")

    args = parser.parse_args()

    if args.cmd == "build":
        cmd_build(args.limit)
    elif args.cmd == "sync":
        cmd_sync()
    elif args.cmd == "search":
        cmd_search(args.query, args.k, args.year)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "delete":
        cmd_delete()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
