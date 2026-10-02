# ============================================================
# start.ps1 - Start the Advanced RAG System
# Run this script from the project folder:
#   PowerShell -ExecutionPolicy Bypass -File start.ps1
# ============================================================

Write-Host "Starting Advanced RAG System..." -ForegroundColor Cyan

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

# Set environment variables for this session
$env:OLLAMA_MODEL = "llama3.2"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "LLM: Ollama (llama3.2) - local, no API key needed" -ForegroundColor Green
Write-Host "Embeddings: sentence-transformers (cached locally)" -ForegroundColor Green
Write-Host "Vector DB: ChromaDB (local)" -ForegroundColor Green
Write-Host ""
Write-Host "Starting server at http://localhost:8000 ..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

python -m uvicorn src.api.main:app --port 8000
