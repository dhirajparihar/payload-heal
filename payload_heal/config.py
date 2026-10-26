"""Configuration settings for payload-heal self-healing models."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

DEFAULT_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3",
    "gemini": "gemini-1.5-flash",
}


@dataclass
class HealConfig:
    """Configuration options for self-healing Pydantic models.

    Attributes:
        enabled: Master toggle for payload healing.
        enable_heuristics: Enables Tier-1 zero-LLM heuristic matcher.
        enable_ai: Enables Tier-2 AI skeleton healer.
        llm_provider: Provider name ('openai', 'groq', 'ollama', 'gemini').
        model: Specific LLM model identifier (defaults based on provider).
        api_key: Provider API key (falls back to environment variables).
        api_base: Custom base URL for the OpenAI-compatible endpoint.
        lockfile_path: Path to the local .payload_heal_lock.json cache.
        webhook_url: URL for schema drift notifications (Slack/Discord/generic).
        timeout_seconds: Network request timeout for LLM calls.
    """

    enabled: bool = True
    enable_heuristics: bool = True
    enable_ai: bool = True
    llm_provider: str = "openai"
    model: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    model_cache_id: str | None = None
    lockfile_path: Path = field(default_factory=lambda: Path(".payload_heal_lock.json"))
    webhook_url: str | None = None
    allowed_webhook_hosts: list[str] | None = None
    allow_internal_webhooks: bool = False
    allowed_ai_hosts: list[str] | None = None
    allow_insecure_ai_endpoints: bool = False
    timeout_seconds: float = 8.0

    def clone(self) -> HealConfig:
        """Returns an independent shallow/deep copy of this configuration."""
        import copy
        return copy.deepcopy(self)

    def __post_init__(self) -> None:
        if isinstance(self.lockfile_path, str):
            self.lockfile_path = Path(self.lockfile_path)

        provider_lower = self.llm_provider.lower().strip()
        self.llm_provider = provider_lower

        # Default model based on provider
        if not self.model:
            self.model = DEFAULT_PROVIDER_MODELS.get(provider_lower, "gpt-4o-mini")

        # Resolve API base URL
        if not self.api_base:
            self.api_base = DEFAULT_PROVIDER_BASE_URLS.get(
                provider_lower,
                "https://api.openai.com/v1",
            )

        # Validate AI endpoint security & trust boundary
        validate_ai_endpoint(
            self.api_base,
            self.allowed_ai_hosts,
            self.allow_insecure_ai_endpoints,
        )

        # Resolve API key from environment if not explicitly set
        if not self.api_key:
            self.api_key = (
                os.getenv("PAYLOAD_HEAL_API_KEY")
                or self._resolve_provider_env_key(provider_lower)
            )

        # Resolve Webhook URL from environment if not explicitly set
        if not self.webhook_url:
            self.webhook_url = os.getenv("PAYLOAD_HEAL_WEBHOOK_URL")

    @staticmethod
    def _resolve_provider_env_key(provider: str) -> str | None:
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        if provider == "groq":
            return os.getenv("GROQ_API_KEY")
        if provider == "gemini":
            return os.getenv("GEMINI_API_KEY")
        if provider == "ollama":
            return os.getenv("OLLAMA_API_KEY") or "ollama"
        return None


def validate_ai_endpoint(
    api_base: str | None,
    allowed_ai_hosts: list[str] | None = None,
    allow_insecure_ai_endpoints: bool = False,
) -> None:
    """Validates the AI api_base endpoint against URL scheme and host restrictions."""
    if not api_base:
        return

    from urllib.parse import urlsplit

    parsed_base = urlsplit(api_base)
    scheme = (parsed_base.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Invalid AI api_base scheme '{scheme}': must be http or https")

    host = (parsed_base.hostname or "").lower()
    # Non-local endpoints must use HTTPS unless allow_insecure_ai_endpoints is explicitly True
    if scheme == "http" and host not in ("localhost", "127.0.0.1", "::1") and not allow_insecure_ai_endpoints:
        raise ValueError(
            f"Insecure HTTP AI endpoint '{api_base}' is disallowed for non-local hosts. "
            "Use HTTPS or set allow_insecure_ai_endpoints=True if explicitly required."
        )

    # Check AI host allowlist if configured
    if allowed_ai_hosts:
        matched = any(
            host == h.lower() or host.endswith(f".{h.lower()}")
            for h in allowed_ai_hosts
        )
        if not matched:
            raise ValueError(f"AI endpoint host '{host}' is not in configured allowed_ai_hosts")
