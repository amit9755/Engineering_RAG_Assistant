# ============================================================
# src/retrieval/reranker_offline.py
# SSL-safe cross-encoder loader for corporate networks
# ============================================================
import os
import ssl

def disable_ssl_for_hf():
    """
    Disable SSL verification AND force offline mode for HuggingFace.

    Learning Note:
        On corporate networks with a proxy, SSL verification fails.
        Setting HF_HUB_OFFLINE=1 forces the library to use the local
        cache only, skipping all network calls. The model was already
        downloaded once so this is safe and fast.
    """
    # Force offline mode - use cached models only, no network calls
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Also disable SSL in case any network call slips through
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    # Patch ssl context
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
