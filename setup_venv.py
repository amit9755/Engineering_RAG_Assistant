"""
setup_venv.py - Install all required packages into the .venv
Run this script with the VENV Python:
    .venv\Scripts\python.exe setup_venv.py
"""
import subprocess
import sys

packages = [
    "langchain>=0.3.0",
    "langchain-huggingface",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-text-splitters>=1.0.0",
    "langgraph>=0.2.14",
    "litellm>=1.44.1",
    "chromadb>=0.5.3",
    "sentence-transformers",
    "rank-bm25",
    "pypdf",
    "python-docx",
]

print(f"Installing {len(packages)} packages into: {sys.executable}")
print("-" * 60)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install"] + packages,
    check=False
)

if result.returncode == 0:
    print("\nAll packages installed successfully!")
    print("Now run: python -m uvicorn src.api.main:app --reload --port 8000")
else:
    print("\nSome packages failed. Check errors above.")
