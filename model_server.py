#!/usr/bin/env python3
"""Embedding & Reranker API Server for Zotero Research Assistant.

Usage:
    python model_server.py [--port 8765] [--reranker jina|qwen|mmarco]

Endpoints:
    POST /embed     {"texts": [...], "mode": "query"|"document"}
    POST /rerank    {"query": "...", "docs": [...], "top_k": 10}
    GET  /health    Health check

Rerankers:
    jina   - jina-reranker-v3-mlx (MLX native, fastest on Apple Silicon)
    qwen   - Qwen3-Reranker-0.6B (CPU)
    mmarco - mmarco-mMiniLMv2-L12-H384-v1 (CPU, default)
"""
import os, sys, json, argparse
from typing import List
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

app = Flask(__name__)

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = None
RERANKER_TYPE = None

_embed_model = None
_embed_tokenizer = None
_reranker = None
_reranker_tokenizer = None
_reranker_config = None
_device = None

def get_device():
    global _device
    if _device is None:
        if torch.backends.mps.is_available():
            _device = "mps"
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
        print(f"[server] Using device: {_device}")
    return _device

# ===== Embedding (Qwen3-Embedding official method) =====

def last_token_pool(last_hidden_states, attention_mask):
    """Official Qwen3-Embedding pooling: use last token."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_query_instruct(query: str) -> str:
    """Add instruction prefix for queries (official format)."""
    task = "Given a web search query, retrieve relevant passages that answer the query"
    return f"Instruct: {task}\nQuery:{query}"

def load_embed_model():
    global _embed_model, _embed_tokenizer
    if _embed_model is None:
        from transformers import AutoModel, AutoTokenizer
        print(f"[server] Loading embedding model: {EMBED_MODEL}")
        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, padding_side='left', trust_remote_code=True)
        _embed_model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        _embed_model = _embed_model.to(get_device())
        if get_device() != "mps":
            _embed_model = _embed_model.half()
        _embed_model.eval()
        print(f"[server] Embedding model loaded (dim={_embed_model.config.hidden_size})")
    return _embed_model, _embed_tokenizer

def get_embeddings(texts: List[str], mode: str = "document") -> List[List[float]]:
    """Generate embeddings using official Qwen3-Embedding method."""
    model, tokenizer = load_embed_model()

    # Add instruction for queries only (official recommendation)
    if mode == "query":
        texts = [get_query_instruct(t) for t in texts]

    embeddings = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # Official: last_token_pool instead of mean pooling
            batch_emb = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            # Official: L2 normalize
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
            embeddings.extend(batch_emb.cpu().float().tolist())

    return embeddings

# ===== Reranker =====

JINA_MODEL_DIR = None

def load_reranker():
    global _reranker, _reranker_tokenizer, _reranker_config, JINA_MODEL_DIR
    if _reranker is None:
        if RERANKER_TYPE == "jina":
            print(f"[server] Loading Jina reranker: {RERANK_MODEL}")
            from huggingface_hub import snapshot_download
            JINA_MODEL_DIR = snapshot_download(RERANK_MODEL)
            import sys
            sys.path.insert(0, JINA_MODEL_DIR)
            prev_cwd = os.getcwd()
            os.chdir(JINA_MODEL_DIR)
            from rerank import MLXReranker
            _reranker = MLXReranker()
            os.chdir(prev_cwd)
            _reranker_tokenizer = None
            print("[server] Jina reranker loaded (MLX)")
        elif RERANKER_TYPE == "qwen":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[server] Loading Qwen reranker: {RERANK_MODEL}")
            _reranker_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side='left', trust_remote_code=True)
            _reranker = AutoModelForCausalLM.from_pretrained(RERANK_MODEL, trust_remote_code=True)
            # Force CPU for Qwen reranker (MPS has issues)
            _reranker = _reranker.to("cpu").half()
            _reranker.eval()

            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            _reranker_config = {
                "prefix_tokens": _reranker_tokenizer.encode(prefix, add_special_tokens=False),
                "suffix_tokens": _reranker_tokenizer.encode(suffix, add_special_tokens=False),
                "token_true_id": _reranker_tokenizer.convert_tokens_to_ids("yes"),
                "token_false_id": _reranker_tokenizer.convert_tokens_to_ids("no"),
                "max_length": 8192,
            }
            print("[server] Qwen reranker loaded (CPU)")
        else:
            from sentence_transformers import CrossEncoder
            print(f"[server] Loading CrossEncoder reranker: {RERANK_MODEL}")
            _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            _reranker_tokenizer = None
            print("[server] CrossEncoder reranker loaded (CPU)")
    return _reranker, _reranker_tokenizer

def qwen_rerank(query: str, docs: List[str], top_k: int = 10) -> List[dict]:
    model, tokenizer = load_reranker()
    cfg = _reranker_config

    instruction = "Given a query, retrieve relevant passages that answer the query"

    def format_pair(q, doc):
        return f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {doc}"

    pairs = [format_pair(query, doc[:2000]) for doc in docs]

    inputs = tokenizer(pairs, padding=False, truncation=True,
                       max_length=cfg["max_length"] - len(cfg["prefix_tokens"]) - len(cfg["suffix_tokens"]),
                       return_attention_mask=False)

    for i, ids in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = cfg["prefix_tokens"] + ids + cfg["suffix_tokens"]

    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=cfg["max_length"])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        true_logits = logits[:, cfg["token_true_id"]]
        false_logits = logits[:, cfg["token_false_id"]]
        stacked = torch.stack([false_logits, true_logits], dim=1)
        probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = probs[:, 1].exp().cpu().tolist()

    results = sorted(zip(range(len(docs)), scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"index": idx, "score": score} for idx, score in results]

def mmarco_rerank(query: str, docs: List[str], top_k: int = 10) -> List[dict]:
    model, _ = load_reranker()
    pairs = [[query, doc[:2000]] for doc in docs]
    scores = model.predict(pairs).tolist()
    results = sorted(zip(range(len(docs)), scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"index": idx, "score": score} for idx, score in results]

def jina_rerank(query: str, docs: List[str], top_k: int = 10) -> List[dict]:
    model, _ = load_reranker()
    truncated_docs = [doc[:2000] for doc in docs]
    results = model.rerank(query, truncated_docs, top_n=top_k)
    return [{"index": r["index"], "score": r["relevance_score"]} for r in results]

# ===== API Routes =====

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": get_device(), "reranker": RERANKER_TYPE})

@app.route("/embed", methods=["POST"])
def embed():
    data = request.json
    texts = data.get("texts", [])
    mode = data.get("mode", "document")

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    embeddings = get_embeddings(texts, mode)
    return jsonify({"embeddings": embeddings, "dim": len(embeddings[0]) if embeddings else 0})

@app.route("/rerank", methods=["POST"])
def rerank():
    data = request.json
    query = data.get("query", "")
    docs = data.get("docs", [])
    top_k = data.get("top_k", 10)

    if not query or not docs:
        return jsonify({"error": "Query and docs required"}), 400

    if RERANKER_TYPE == "jina":
        results = jina_rerank(query, docs, top_k)
    elif RERANKER_TYPE == "qwen":
        results = qwen_rerank(query, docs, top_k)
    else:
        results = mmarco_rerank(query, docs, top_k)

    return jsonify({"results": results})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preload", action="store_true", help="Preload models on startup")
    parser.add_argument("--reranker", choices=["jina", "qwen", "mmarco"], default="jina",
                        help="Reranker: jina (MLX), qwen (CPU), or mmarco (CPU)")
    args = parser.parse_args()

    RERANKER_TYPE = args.reranker
    if RERANKER_TYPE == "jina":
        RERANK_MODEL = "jinaai/jina-reranker-v3-mlx"
    elif RERANKER_TYPE == "qwen":
        RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
    else:
        RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    print(f"[server] Starting on http://127.0.0.1:{args.port}")
    print(f"[server] Reranker: {RERANKER_TYPE} ({RERANK_MODEL})")

    if args.preload:
        print("[server] Preloading models...")
        load_embed_model()
        load_reranker()
        print("[server] Models ready!")

    app.run(host="127.0.0.1", port=args.port, threaded=False)
