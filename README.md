# Zotero Research Assistant

Multimodal semantic search over your Zotero library using **Qdrant + Jina-embeddings-v4**.

## Architecture

```
Zotero 8 (localhost:23119)
       │
       ▼
   PDF Files
       │
       ▼ PyMuPDF (150 DPI, max 15 pages/paper)
       │
  Page Images
       │
       ▼ Jina-embeddings-v4 (multimodal)
       │
  ┌────┴────┐
  │         │
Dense    Multi-vector
2048d    128d × N tokens
  │         │
  └────┬────┘
       ▼
    Qdrant (local)
       │
       ▼ Two-stage retrieval
       │
   Results
```

## Retrieval Pipeline

### Stage 1: Dense Prefetch
- Query encoded to single 2048d vector
- Fast cosine similarity search
- Returns top 100 candidates

### Stage 2: MaxSim Rerank
- Query encoded to multi-vector (128d × N tokens)
- Each query token computes max similarity across all document patches
- Sum of max similarities = final score
- Returns top K results

This **late interaction** approach captures fine-grained token-level matching while maintaining efficiency through the two-stage design.

## Key Components

| Component | Purpose |
|-----------|---------|
| `workspace.py` | Main CLI - build, sync, search, status, delete |
| `zotero-query.py` | Metadata queries via Zotero API |
| Jina-v4 | Multimodal encoder (text + image, 4B params) |
| Qdrant | Vector DB with MaxSim support |
| PyMuPDF | PDF to image conversion |

## Data Flow

### Indexing (`build`)
1. Fetch papers with PDFs from Zotero API
2. Convert each PDF to page images (150 DPI)
3. Encode images with Jina-v4:
   - `dense`: Single 2048d vector per page
   - `multivector`: 128d × N tokens per page
4. Store in Qdrant with payload (title, authors, year, page_num)

### Search (`search`)
1. Encode query text with Jina-v4 (both dense + multi-vector)
2. Prefetch: Dense similarity → top 100 pages
3. Rerank: MaxSim on multi-vectors → top K×3 pages
4. Deduplicate by paper (keep best page per paper)
5. Return top K papers with scores

## Why This Approach?

**vs. Text-only RAG:**
- Captures figures, tables, equations that text extraction misses
- No OCR errors from complex layouts

**vs. Separate text/image indexes:**
- Single unified embedding space
- Simpler architecture, easier to maintain

**vs. ColPali/ColQwen:**
- Jina-v4 supports both single-vector (fast) and multi-vector (precise)
- Better macOS MPS compatibility
- Proven production quality

## Storage

```
~/.local/share/zotero-qdrant/
├── qdrant_db/          # Vector database
│   ├── collection/     # Indexed vectors
│   └── meta.json       # DB metadata
└── meta.json           # Index metadata (keys, version)
```

## Performance

On M4 Max with MPS acceleration:
- Model load: ~25s
- Page encoding: ~20s/page (varies with content)
- Search latency: <500ms (100 prefetch + MaxSim rerank)
- Storage: ~4MB per 10 papers (varies with page count)

## Dependencies

```
torch
transformers
qdrant-client
pymupdf
pillow
requests
tqdm
```
