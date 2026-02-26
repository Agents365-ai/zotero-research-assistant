#!/usr/bin/env python3
"""Test TomoroAI/tomoro-colqwen3-embed-4b on macOS M4 Max.

Usage:
    python test_colqwen3_macos.py              # Basic test with sample image
    python test_colqwen3_macos.py --pdf FILE   # Test with a PDF file
    python test_colqwen3_macos.py --benchmark  # Run performance benchmark
"""
import sys, os, time, argparse
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model():
    import torch
    from transformers import AutoModel, AutoProcessor

    MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
    DEVICE = get_device()
    DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

    print(f"[info] Loading model on {DEVICE} with {DTYPE}...")
    start = time.time()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        max_num_visual_tokens=1280,
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(DEVICE)
    model.requires_grad_(False)

    print(f"[info] Model loaded in {time.time()-start:.1f}s")
    return model, processor, DEVICE, DTYPE

def encode_queries(model, processor, texts, device, dtype, batch_size=8):
    import torch
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = processor.process_texts(texts=texts[start:start+batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            out = model(**batch)
            vecs = out.embeddings.to(dtype).cpu()
        outputs.extend(vecs)
    return outputs

def encode_images(model, processor, images, device, dtype, batch_size=4):
    import torch
    outputs = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start+batch_size]
        features = processor.process_images(images=batch_imgs)
        features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
        with torch.inference_mode():
            out = model(**features)
            vecs = out.embeddings.to(dtype).cpu()
        outputs.extend(vecs)
    return outputs

def pdf_to_images(pdf_path, max_pages=5):
    try:
        import fitz
    except ImportError:
        print("[error] PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(1)

    from PIL import Image
    import io

    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    doc.close()
    print(f"[info] Extracted {len(images)} pages from {Path(pdf_path).name}")
    return images

def test_basic():
    from PIL import Image
    import numpy as np

    print("\n=== Basic Test ===")
    model, processor, device, dtype = load_model()

    img_array = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    test_image = Image.fromarray(img_array)

    queries = [
        "What is shown in this document?",
        "Find information about cancer treatment",
    ]

    print(f"[info] Encoding {len(queries)} queries...")
    start = time.time()
    q_embs = encode_queries(model, processor, queries, device, dtype)
    print(f"[info] Query encoding: {time.time()-start:.2f}s")

    print(f"[info] Encoding 1 image...")
    start = time.time()
    i_embs = encode_images(model, processor, [test_image], device, dtype)
    print(f"[info] Image encoding: {time.time()-start:.2f}s")

    scores = processor.score_multi_vector(q_embs, i_embs)
    print(f"\n[result] MaxSim scores:")
    for q, s in zip(queries, scores[0].tolist()):
        print(f"  '{q[:40]}' -> {s:.4f}")

    print(f"\n[info] Embedding shapes:")
    print(f"  Query: {q_embs[0].shape} (tokens x dim)")
    print(f"  Image: {i_embs[0].shape} (patches x dim)")

    return True

def test_pdf(pdf_path):
    print(f"\n=== PDF Test: {Path(pdf_path).name} ===")

    if not Path(pdf_path).exists():
        print(f"[error] File not found: {pdf_path}")
        return False

    model, processor, device, dtype = load_model()
    images = pdf_to_images(pdf_path, max_pages=3)

    queries = [
        "What is the main finding of this paper?",
        "Show me the methods section",
        "Where are the results and figures?",
    ]

    print(f"[info] Encoding queries...")
    q_embs = encode_queries(model, processor, queries, device, dtype)

    print(f"[info] Encoding {len(images)} pages...")
    start = time.time()
    i_embs = encode_images(model, processor, images, device, dtype)
    elapsed = time.time() - start
    print(f"[info] Page encoding: {elapsed:.2f}s ({elapsed/len(images):.2f}s/page)")

    scores = processor.score_multi_vector(q_embs, i_embs)

    print(f"\n[result] Query-Page similarity matrix:")
    header = "Query".ljust(45) + " | " + " | ".join([f"Page {i+1}" for i in range(len(images))])
    print(header)
    print("-" * len(header))
    for q, row in zip(queries, scores.tolist()):
        scores_str = " | ".join([f"{s:>6.3f}" for s in row])
        print(f"{q:<45} | {scores_str}")

    return True

def benchmark():
    from PIL import Image
    import numpy as np

    print("\n=== Performance Benchmark ===")
    model, processor, device, dtype = load_model()

    n_images = 10
    images = [Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)) for _ in range(n_images)]
    queries = [f"Test query number {i}" for i in range(5)]

    print("[info] Warming up...")
    _ = encode_queries(model, processor, queries[:1], device, dtype)
    _ = encode_images(model, processor, images[:1], device, dtype)

    print(f"[info] Benchmarking {len(queries)} queries...")
    start = time.time()
    for _ in range(3):
        q_embs = encode_queries(model, processor, queries, device, dtype)
    q_time = (time.time() - start) / 3

    print(f"[info] Benchmarking {n_images} images...")
    start = time.time()
    i_embs = encode_images(model, processor, images, device, dtype)
    i_time = time.time() - start

    print(f"\n[result] Benchmark Results (M4 Max + MPS):")
    print(f"  Query encoding:  {q_time:.2f}s for {len(queries)} queries ({q_time/len(queries)*1000:.0f}ms/query)")
    print(f"  Image encoding:  {i_time:.2f}s for {n_images} images ({i_time/n_images*1000:.0f}ms/image)")
    print(f"  Estimated for 1000 pages: {i_time/n_images*1000:.0f}s ({i_time/n_images*1000/60:.1f} min)")

    return True

def main():
    parser = argparse.ArgumentParser(description="Test ColQwen3 on macOS")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to test")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("TomoroAI/tomoro-colqwen3-embed-4b - macOS Test")
    print("=" * 60)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {get_device()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if args.pdf:
        success = test_pdf(args.pdf)
    elif args.benchmark:
        success = benchmark()
    else:
        success = test_basic()

    status = "PASSED" if success else "FAILED"
    print("\n" + "=" * 60)
    print(f"Test {status}")
    print("=" * 60)

if __name__ == "__main__":
    main()
