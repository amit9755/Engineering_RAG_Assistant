"""
One switch for outbound network access: ALLOW_EXTERNAL_NETWORK.

When false, the app only contacts internal hosts - names under INTERNAL_DOMAINS
(e.g. bitbucket.sw.nxp.com), localhost (Ollama) and private IP addresses - and
everything that would leave the company network is blocked with a clear error:
cloud LLM providers, Hugging Face downloads, LiteLLM's price-list fetch, Chroma
and LiteLLM telemetry, Langfuse cloud tracing, spaCy model downloads, Google
Cloud Storage, and Bitbucket / Jira Cloud.
"""

import ipaddress
import os
from urllib.parse import urlparse

from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class ExternalNetworkBlocked(Exception):
    """Raised instead of contacting a host outside the company network."""


def external_allowed() -> bool:
    return bool(settings.allow_external_network)


def internal_domains() -> list:
    return [d.strip().lower().lstrip(".") for d in settings.internal_domains.split(",") if d.strip()]


def is_internal_host(host: str) -> bool:
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return False
    if host in ("localhost", "host.docker.internal") or "." not in host:
        return True  # localhost and single-label intranet names
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        pass
    return any(host == d or host.endswith("." + d) for d in internal_domains())


def is_internal_url(url: str) -> bool:
    return is_internal_host(urlparse(url).hostname or "")


def check_url(url: str, purpose: str = "this request") -> None:
    """Raise ExternalNetworkBlocked if url is outside the allowed network."""
    if external_allowed() or is_internal_url(url):
        return
    host = urlparse(url).hostname or url
    raise ExternalNetworkBlocked(
        f"Blocked {purpose}: {host} is outside the company network. External calls are disabled "
        f"(ALLOW_EXTERNAL_NETWORK=false; internal domains: {', '.join(internal_domains()) or 'none'}).")


def apply() -> None:
    """Configure third-party libraries before they make network calls. Call once at startup."""
    if external_allowed():
        logger.info("network_policy", external_calls="allowed")
        return
    os.environ["HF_HUB_OFFLINE"] = "1"                  # Hugging Face model downloads
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"  # LiteLLM price list from GitHub
    os.environ["ANONYMIZED_TELEMETRY"] = "False"         # ChromaDB telemetry
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import litellm
        litellm.telemetry = False
    except ImportError:
        pass
    logger.info("network_policy", external_calls="blocked", internal_domains=internal_domains())
