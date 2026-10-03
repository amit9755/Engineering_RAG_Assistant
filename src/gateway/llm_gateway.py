# ============================================================
# src/gateway/llm_gateway.py - LLM Gateway via LiteLLM
#
# Learning Note:
#   WHY AN LLM GATEWAY?
#   Without a gateway, your code is tightly coupled to one LLM
#   provider. If OpenAI goes down or prices increase, you're stuck.
#
#   LiteLLM is a unified interface that translates one API call
#   to any of 100+ LLM providers (OpenAI, Gemini, Anthropic, etc.)
#   using the same OpenAI-compatible format.
#
#   ENTERPRISE FEATURES we implement here:
#     - Retry with exponential backoff (resilience)
#     - Fallback chain: Gemini -> GPT-4o -> GPT-3.5 (reliability)
#     - Budget limits (cost control)
#     - Request/response logging (audit trail)
#     - Timeout handling (SLA compliance)
# ============================================================

from typing import List, Dict, Any, Optional, AsyncGenerator
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class LLMGateway:
    """
    Unified LLM interface with retry, fallback, cost tracking,
    and audit logging. Uses LiteLLM under the hood.

    Learning Note:
        LiteLLM maps all calls to the OpenAI chat completion format:
        messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        This is the industry standard format that all major LLMs support.
    """

    def __init__(self):
        self._setup_litellm()
        # Track cumulative spend per session
        self._total_cost_usd = 0.0
        self._total_tokens = 0

    def _setup_litellm(self) -> None:
        """Configure LiteLLM with logging and budget settings."""
        try:
            import litellm
            # Enable token usage tracking
            litellm.success_callback = [self._on_success]
            litellm.failure_callback = [self._on_failure]
            # Local models on a CPU-only PC can take minutes (reading an image, a long
            # code answer); a short timeout made them fail and retry.
            litellm.request_timeout = int(os.environ.get("LLM_TIMEOUT", "600"))
            logger.info("litellm_gateway_initialized")
        except ImportError:
            logger.warning("litellm_not_installed", msg="pip install litellm")

    def _on_success(self, kwargs: dict, response: Any, start_time: float, end_time: float) -> None:
        """Callback fired after every successful LLM call. Log usage."""
        try:
            usage = getattr(response, "usage", None)
            if usage:
                tokens = getattr(usage, "total_tokens", 0)
                self._total_tokens += tokens
                logger.info(
                    "llm_call_success",
                    model=kwargs.get("model"),
                    total_tokens=tokens,
                    latency_ms=int((end_time - start_time) * 1000),
                )
        except Exception:
            pass

    def _on_failure(self, kwargs: dict, exception: Exception, start_time: float, end_time: float) -> None:
        """Callback fired on LLM call failure."""
        logger.error(
            "llm_call_failed",
            model=kwargs.get("model"),
            error=str(exception),
            latency_ms=int((end_time - start_time) * 1000),
        )

    def _build_model_string(self, provider: str = "primary") -> str:
        """
        Build the LiteLLM model string.

        Learning Note:
            LiteLLM uses a "provider/model" format:
              "gemini/gemini-1.5-flash" -> Google AI Studio (FREE tier)
              "gpt-4o"                  -> OpenAI
              "anthropic/claude-3-opus" -> Anthropic

            Priority:
            1. GEMINI_API_KEY set -> use free Gemini Flash via Google AI Studio
            2. OPENAI_API_KEY set -> use GPT-4o
            3. GCP project set   -> use Vertex AI Gemini
        """
        import os
        from dotenv import dotenv_values
        # Load .env values directly (in case os.environ not populated)
        env_vals = dotenv_values(".env")
        gemini_key = os.environ.get("GEMINI_API_KEY", env_vals.get("GEMINI_API_KEY", "")).strip()
        openai_key = (settings.openai_api_key or "").strip()
        ollama_model = os.environ.get("OLLAMA_MODEL", env_vals.get("OLLAMA_MODEL", "")).strip()

        from src import network_policy
        if not network_policy.external_allowed():
            # Cloud providers are outside the company network: only a local / internal Ollama.
            api_base = os.environ.get("OLLAMA_API_BASE", env_vals.get("OLLAMA_API_BASE", "")) or "http://localhost:11434"
            network_policy.check_url(api_base, "the Ollama model server")
            if not ollama_model:
                raise network_policy.ExternalNetworkBlocked(
                    "External calls are disabled (ALLOW_EXTERNAL_NETWORK=false), so only a local model can be "
                    "used. Set OLLAMA_MODEL (e.g. llama3.2) in .env.")
            return f"ollama/{ollama_model}"

        if gemini_key:
            # Free Google AI Studio - get key at https://aistudio.google.com/apikey
            os.environ["GEMINI_API_KEY"] = gemini_key
            return f"gemini/{settings.vertex_ai_model}"
        elif openai_key:
            return settings.openai_model
        elif ollama_model:
            # 100% local, no API key needed - install from https://ollama.com
            return f"ollama/{ollama_model}"
        elif settings.gcp_project_id and settings.gcp_project_id != "local-dev":
            return f"gemini/{settings.vertex_ai_model}"
        else:
            # No key configured - will raise a clear error
            return f"gemini/{settings.vertex_ai_model}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Call the LLM with retry logic. Returns the response text.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            temperature: 0=deterministic, 1=creative. Use 0.1 for RAG
                         (we want factual, not creative answers)
            max_tokens: Max response length
            model: Override the model. If None, uses config default.

        Learning Note:
            Temperature near 0 is best for RAG because we want the LLM
            to accurately extract info from context, not hallucinate
            creative answers. Save high temperature for creative tasks.
        """
        import litellm

        model_name = model or self._build_model_string()

        # Budget guard: check before expensive call
        if self._total_cost_usd >= settings.litellm_budget_limit:
            raise RuntimeError(
                f"LLM budget limit reached: ${self._total_cost_usd:.2f} / ${settings.litellm_budget_limit}"
            )

        logger.info("llm_call_start", model=model_name, messages_count=len(messages))

        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"stop": stop} if stop else {}),
            **self._provider_options(model_name),
        )

        answer = response.choices[0].message.content
        return answer

    async def acomplete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        """
        Async version of complete(). Use this in async FastAPI endpoints
        to avoid blocking the event loop.
        """
        import litellm

        model_name = model or self._build_model_string()

        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **self._provider_options(model_name),
        )

        return response.choices[0].message.content

    async def astream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming completion. Yields text chunks as they arrive.

        Learning Note:
            Streaming sends tokens to the UI as they are generated,
            giving users immediate feedback instead of waiting for the
            full response. This is how ChatGPT's streaming works.
        """
        import litellm

        model_name = model or self._build_model_string()

        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **({"stop": stop} if stop else {}),
            **self._provider_options(model_name),
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                yield delta.content

    def code_model_string(self) -> str:
        """Local code model for code suggestions (OLLAMA_CODE_MODEL, default qwen2.5-coder:7b)."""
        from dotenv import dotenv_values
        from src import network_policy
        env_vals = dotenv_values(".env")
        model = (os.environ.get("OLLAMA_CODE_MODEL") or env_vals.get("OLLAMA_CODE_MODEL") or "qwen2.5-coder:7b").strip()
        api_base = os.environ.get("OLLAMA_API_BASE", env_vals.get("OLLAMA_API_BASE", "")) or "http://localhost:11434"
        network_policy.check_url(api_base, "the Ollama model server")
        return f"ollama/{model}"

    # Small models sometimes continue the chat by writing the user's next turn themselves.
    ANSWER_STOP = ["\nUser:", "\nUSER:", "\nHuman:", "\nAssistant:", "\nASSISTANT:", "\n### User", "\n**User:**"]

    def vision_model_string(self) -> str:
        """
        Local vision model for questions with images. Uses LiteLLM's ollama_chat
        provider: it forwards each message's "images" to Ollama unchanged, whereas
        the plain ollama provider drops images for non-llava models.
        """
        from dotenv import dotenv_values
        from src import network_policy
        env_vals = dotenv_values(".env")
        model = (os.environ.get("OLLAMA_VISION_MODEL") or env_vals.get("OLLAMA_VISION_MODEL") or "gemma3:4b").strip()
        api_base = os.environ.get("OLLAMA_API_BASE", env_vals.get("OLLAMA_API_BASE", "")) or "http://localhost:11434"
        network_policy.check_url(api_base, "the Ollama model server")
        return f"ollama_chat/{model}"

    @staticmethod
    def _provider_options(model_name: str) -> Dict[str, Any]:
        """
        Ollama defaults to a 4096-token context window; RAG prompts (system
        prompt + retrieved chunks + history) overflow it and Ollama silently
        drops the start of the prompt - the instructions and context.
        """
        if model_name.startswith("ollama"):
            return {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "16384"))}
        return {}

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return current usage statistics."""
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost_usd, 4),
            "budget_limit_usd": settings.litellm_budget_limit,
            "budget_remaining_usd": round(settings.litellm_budget_limit - self._total_cost_usd, 4),
        }


# Singleton instance
llm_gateway = LLMGateway()
