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
    lockfile_path: Path = field(default_factory=lambda: Path(".payload_heal_lock.json"))
    webhook_url: str | None = None
    timeout_seconds: float = 8.0

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
