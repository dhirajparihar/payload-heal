"""Payload-Heal: Production-ready self-healing Pydantic models for resilient JSON/GraphQL ingestion."""

from __future__ import annotations

from payload_heal.ai_healer import (
    build_user_prompt,
    extract_json_from_response,
    heal_with_llm,
)
from payload_heal.config import (
    DEFAULT_PROVIDER_BASE_URLS,
    DEFAULT_PROVIDER_MODELS,
    HealConfig,
)
from payload_heal.heuristics import (
    find_heuristic_matches,
    is_type_compatible,
    normalize_key,
    score_field_candidate,
    tokenize_key,
)
from payload_heal.lockfile import (
    DEFAULT_LOCKFILE_PATH,
    LockFileManager,
    get_mapping,
    invalidate_field,
    save_mapping,
    set_default_lockfile_path,
)
from payload_heal.model import (
    HealableModel,
    auto_heal_dict,
    get_field_type_str,
    heal_json,
    heal_payload,
)
from payload_heal.notifier import send_drift_notification
from payload_heal.skeleton import (
    extract_json_skeleton,
    format_sample_preview,
    get_by_dot_path,
    is_luhn_valid,
    mask_pii,
    parse_dot_path,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_LOCKFILE_PATH",
    "DEFAULT_PROVIDER_BASE_URLS",
    "DEFAULT_PROVIDER_MODELS",
    "HealConfig",
    "HealableModel",
    "LockFileManager",
    "auto_heal_dict",
    "build_user_prompt",
    "extract_json_from_response",
    "extract_json_skeleton",
    "find_heuristic_matches",
    "format_sample_preview",
    "get_by_dot_path",
    "get_field_type_str",
    "get_mapping",
    "heal_json",
    "heal_payload",
    "heal_with_llm",
    "invalidate_field",
    "is_luhn_valid",
    "is_type_compatible",
    "mask_pii",
    "normalize_key",
    "parse_dot_path",
    "save_mapping",
    "score_field_candidate",
    "send_drift_notification",
    "set_default_lockfile_path",
    "tokenize_key",
]
