"""
download_model.py - Pre-download the embedding model to local cache.
Run ONCE before starting the server:
    .venv\Scripts\python.exe download_model.py

After this completes, the server starts in <3 seconds every time.
The model is cached at: C:\Users\<you>\.cache\huggingface\hub\
"""
import sys
print("Downloading sentence-transformers/all-MiniLM-L6-v2 (90MB)...")
print("This is a ONE-TIME download. Future starts will be instant.")
print("-" * 60)

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # Test it works
    vec = model.encode("test sentence")
    print(f"\nModel downloaded and verified! Vector dimension: {len(vec)}")
    print("You can now start the server:")
    print("  python -m uvicorn src.api.main:app --reload --port 8000")
except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)
