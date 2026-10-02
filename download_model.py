"""
download_model.py - Pre-download the local models to the Hugging Face cache.
Run ONCE on a new machine, before starting the server:
    macOS/Linux:  .venv-macos/bin/python download_model.py
    Windows:      .venv\\Scripts\\python.exe download_model.py

The server runs with HF_HUB_OFFLINE=1 and only uses cached models, so both
models must be downloaded here first. Without the re-ranker, retrieval
silently falls back to unranked results and answer quality drops.
"""
import os
import sys

# Allow network access for this script even if .env sets offline mode.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print("1/2 Embedding model: sentence-transformers/all-MiniLM-L6-v2 (~90 MB)")
    vec = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode("test sentence")
    print(f"    OK - vector dimension {len(vec)}")

    print("2/2 Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB)")
    scores = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict(
        [("what is rag", "Retrieval-augmented generation answers from documents"),
         ("what is rag", "The weather is sunny today")])
    print(f"    OK - relevant {scores[0]:.1f} vs irrelevant {scores[1]:.1f}")

    print("\nBoth models are cached. Start the server:")
    print("  python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)
