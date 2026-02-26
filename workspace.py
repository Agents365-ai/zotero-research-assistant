#!/usr/bin/env python3
"""Zotero Research Assistant v2: Qdrant + Jina-v4 multimodal retrieval.

Usage:
  python workspace_v2.py build [--limit N]     Build index from Zotero PDFs
  python workspace_v2.py sync                  Sync new papers
  python workspace_v2.py search <query> [-k N] Search with two-stage retrieval
  python workspace_v2.py status                Show index status
  python workspace_v2.py delete                Delete index
"""
import sys, os, json, argparse, time
from pathlib import Path
from typing import Optional
import urllib.parse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = Path.home() / ".local/share/zotero-qdrant"
META_FILE = DATA_DIR / "meta.json"
COLLECTION = "zotero_papers"

def out(msg): print(f"[zotero-v2] {msg}", flush=True)

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
        children, _ = zotero_api(f"items/{key}/children")
        for c in children:
            if c["data"].get("contentType") == "application/pdf":
                link = c.get("links", {}).get("enclosure", {})
                pdf_path = urllib.parse.unquote(link.get("href", "").replace("file://", ""))
                if pdf_path and os.path.exists(pdf_path):
                    result.append({
                        "key": key, "title": title, "authors": authors,
                        "year": int(year) if year.isdigit() else 0, "pdf": pdf_path
                    })
                    break
    return result, ver

def pdf_to_images(pdf_path, max_pages=15, dpi=150):
    import fitz
    from PIL import Image
    import io
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages: break
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    doc.close()
    return images

# --- Jina-v4 Model ---

_model = None

def load_model():
    global _model
    if _model is not None: return _model
    import torch
    from transformers import AutoModel
    out(f"Loading Jina-v4 on {get_device()}...")
    start = time.time()
    _model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v4",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(get_device())
    _model.requires_grad_(False)
    out(f"Model loaded in {time.time()-start:.1f}s")
    return _model

def encode_images(images, batch_size=4):
    import torch
    model = load_model()
    dense_list, multi_list = [], []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        with torch.inference_mode():
            dense = model.encode_image(batch, task="retrieval")
            multi = model.encode_image(batch, task="retrieval", return_multivector=True)
        dense_list.extend([d.cpu().tolist() for d in dense])
        multi_list.extend([m.cpu().tolist() for m in multi])
    return dense_list, multi_list

def encode_query(text):
    import torch
    model = load_model()
    with torch.inference_mode():
        dense = model.encode_text([text], task="retrieval", prompt_name="query")[0]
        multi = model.encode_text([text], task="retrieval", prompt_name="query", return_multivector=True)[0]
    return dense.cpu().tolist(), multi.cpu().tolist()

# --- Qdrant ---

def get_client():
    from qdrant_client import QdrantClient
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(DATA_DIR / "qdrant_db"))

def create_collection(client):
    from qdrant_client import models
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(size=2048, distance=models.Distance.COSINE),
            "multivector": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
    )
    out(f"Created collection: {COLLECTION}")

def load_meta():
    if META_FILE.exists(): return json.loads(META_FILE.read_text())
    return {"indexed_keys": [], "library_version": 0, "total_pages": 0}

def save_meta(meta):
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, indent=2))

# --- Commands ---

def cmd_build(limit=None):
    from qdrant_client import models
    from tqdm import tqdm

    out("Fetching papers from Zotero...")
    items, ver = get_pdf_items(limit=limit or 500)
    out(f"Found {len(items)} papers with PDFs")
    if not items: return

    client = get_client()
    create_collection(client)

    total_pages = 0
    indexed_keys = []

    for item in tqdm(items, desc="Indexing"):
        try:
            images = pdf_to_images(item["pdf"], max_pages=15)
            if not images: continue

            dense_vecs, multi_vecs = encode_images(images)

            points = []
            for page_num, (dense, multi) in enumerate(zip(dense_vecs, multi_vecs)):
                points.append(models.PointStruct(
                    id=hash(f"{item['key']}_{page_num}") % (2**63),
                    payload={
                        "zotero_key": item["key"],
                        "title": item["title"],
                        "authors": item["authors"],
                        "year": item["year"],
                        "page_num": page_num,
                    },
                    vector={"dense": dense, "multivector": multi},
                ))

            client.upsert(collection_name=COLLECTION, points=points)
            indexed_keys.append(item["key"])
            total_pages += len(images)

        except Exception as e:
            out(f"Error indexing {item['title'][:30]}: {e}")

    meta = {"indexed_keys": indexed_keys, "library_version": ver, "total_pages": total_pages}
    save_meta(meta)
    out(f"Done! Indexed {len(indexed_keys)} papers, {total_pages} pages")

def cmd_sync():
    meta = load_meta()
    if not meta["indexed_keys"]:
        out("No existing index. Run 'build' first.")
        return

    items, ver = get_pdf_items()
    existing = set(meta["indexed_keys"])
    new_items = [it for it in items if it["key"] not in existing]

    if not new_items:
        out("Library up to date.")
        return

    out(f"Found {len(new_items)} new papers")
    from qdrant_client import models
    from tqdm import tqdm

    client = get_client()
    added = 0

    for item in tqdm(new_items, desc="Adding"):
        try:
            images = pdf_to_images(item["pdf"], max_pages=15)
            if not images: continue

            dense_vecs, multi_vecs = encode_images(images)
            points = []
            for page_num, (dense, multi) in enumerate(zip(dense_vecs, multi_vecs)):
                points.append(models.PointStruct(
                    id=hash(f"{item['key']}_{page_num}") % (2**63),
                    payload={
                        "zotero_key": item["key"],
                        "title": item["title"],
                        "authors": item["authors"],
                        "year": item["year"],
                        "page_num": page_num,
                    },
                    vector={"dense": dense, "multivector": multi},
                ))

            client.upsert(collection_name=COLLECTION, points=points)
            meta["indexed_keys"].append(item["key"])
            meta["total_pages"] += len(images)
            added += 1
        except Exception as e:
            out(f"Error: {e}")

    meta["library_version"] = ver
    save_meta(meta)
    out(f"Added {added} papers")

def cmd_search(query: str, top_k: int = 10, year_min: Optional[int] = None):
    from qdrant_client import models

    client = get_client()
    if not client.collection_exists(COLLECTION):
        out("No index found. Run 'build' first.")
        return

    out(f"Searching: {query}")
    dense_q, multi_q = encode_query(query)

    # Build filter
    filter_cond = None
    if year_min:
        filter_cond = models.Filter(
            must=[models.FieldCondition(key="year", range=models.Range(gte=year_min))]
        )

    # Two-stage: dense prefetch (100) -> MaxSim rerank (top_k)
    start = time.time()
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=dense_q,
                using="dense",
                limit=100,
                filter=filter_cond,
            )
        ],
        query=multi_q,
        using="multivector",
        limit=top_k * 3,  # Get more to dedupe by paper
        with_payload=True,
    )

    # Dedupe by paper, keep best page per paper
    seen = {}
    for pt in results.points:
        key = pt.payload["zotero_key"]
        if key not in seen or pt.score > seen[key]["score"]:
            seen[key] = {
                "score": pt.score,
                "title": pt.payload["title"],
                "authors": pt.payload["authors"],
                "year": pt.payload["year"],
                "page": pt.payload["page_num"] + 1,
                "key": key,
            }

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    elapsed = time.time() - start

    print(json.dumps({"query": query, "results": ranked, "time_ms": int(elapsed*1000)},
                     ensure_ascii=False, indent=2))

def cmd_status():
    client = get_client()
    if not client.collection_exists(COLLECTION):
        out("No index found.")
        return

    info = client.get_collection(COLLECTION)
    meta = load_meta()

    print(f"""
=== Zotero-Qdrant Index Status ===
Collection: {COLLECTION}
Papers indexed: {len(meta.get('indexed_keys', []))}
Total pages: {meta.get('total_pages', 0)}
Points in DB: {info.points_count}
Vectors: dense (2048d) + multivector (128d, MaxSim)
Library version: {meta.get('library_version', 0)}
Data path: {DATA_DIR}
""")

def cmd_delete():
    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        out(f"Deleted collection: {COLLECTION}")
    if META_FILE.exists():
        META_FILE.unlink()
        out("Deleted metadata")

def main():
    parser = argparse.ArgumentParser(description="Zotero + Qdrant + Jina-v4")
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
