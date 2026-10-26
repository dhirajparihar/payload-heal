"""Core HealableModel and standalone @heal_json decorator for payload-heal.

Provides drop-in resilience for Pydantic v2 models against upstream JSON/GraphQL
schema drift across 3 tiers:
- Tier 0: 0ms lockfile lookup with automatic re-drift invalidation
- Tier 1: Zero-LLM deterministic heuristic matcher
- Tier 2: AI Skeleton Healer with anti-hallucination type verification
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, model_validator
from pydantic.fields import FieldInfo

from payload_heal.ai_healer import heal_with_llm
from payload_heal.config import HealConfig
from payload_heal.heuristics import find_heuristic_matches, is_type_compatible
from payload_heal.lockfile import LockFileManager
from payload_heal.notifier import send_drift_notification
from payload_heal.skeleton import extract_json_skeleton, get_by_dot_path

logger = logging.getLogger("payload_heal.model")


def get_model_identifier(schema: type[BaseModel] | str) -> str:
    """Returns a unique, stable identifier for a Pydantic model class.

    Combines module and class name (e.g. `myapp.models.Product`)
    to prevent cross-model cached mapping collisions.
    """
    if isinstance(schema, str):
        return schema
    mod = getattr(schema, "__module__", "")
    name = getattr(schema, "__name__", "Model")
    if mod and mod not in ("__main__", "builtins"):
        return f"{mod}.{name}"
    return name


def get_field_type_str(field_info: FieldInfo) -> str:
    """Extracts a clean, canonical type annotation string from a Pydantic FieldInfo."""
    ann = field_info.annotation
    if ann is None:
        return "Any"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann).replace("typing.", "")


def auto_heal_dict(
    data: dict[str, Any],
    model_name: str,
    model_fields: dict[str, FieldInfo],
    config: HealConfig,
) -> dict[str, Any]:
    """Applies the 3-tier healing pipeline to an incoming dictionary payload.

    Tiers:
    0. Fast Path & Lockfile cache lookup (with type verification)
    1. Heuristic matcher (with type verification)
    2. AI Skeleton Healer (LLM query with strict type-safety verification)

    Returns:
        The updated dictionary containing healed field values.
    """
    if not config.enabled:
        return data

    def is_present(name: str, f: FieldInfo) -> bool:
        alias = getattr(f, "alias", None)
        return name in data or (bool(alias) and alias in data)

    # Fast Path (0ms overhead): Only fields that are required for validation trigger healing
    missing_required = {
        name: f for name, f in model_fields.items()
        if not is_present(name, f) and f.is_required()
    }
    if not missing_required:
        return data

    missing_fields = dict(missing_required)

    healed_data = dict(data)
    healed_notifications: dict[str, tuple[str, str]] = {}
    lock_manager = LockFileManager(config.lockfile_path)

    # --------------------------------------------------------------------------
    # Step 1: Tier 0 Lockfile Lookup & Re-Drift Invalidation
    # --------------------------------------------------------------------------
    cached_mappings = lock_manager.get_mapping(model_name)
    for field_name in list(missing_fields.keys()):
        if field_name in cached_mappings:
            source_path = cached_mappings[field_name]
            found, val = get_by_dot_path(data, source_path)
            if found:
                # Critical Safety Check: verify type compatibility for cached values
                expected_type = get_field_type_str(missing_fields[field_name])
                actual_type = type(val).__name__ if val is not None else "null"
                if is_type_compatible(expected_type, actual_type, val):
                    healed_data[field_name] = val
                else:
                    logger.warning(
                        "Cached path '%s' for field '%s' contains incompatible value %r (expected '%s', got '%s'). Invalidating cache.",
                        source_path,
                        field_name,
                        val,
                        expected_type,
                        actual_type,
                    )
                    lock_manager.invalidate_field(model_name, field_name)
            else:
                # Upstream API drifted a second time! Invalidate stale cache
                logger.info(
                    "Cached path '%s' for field '%s' no longer exists in payload. Invalidating stale lockfile entry.",
                    source_path,
                    field_name,
                )
                lock_manager.invalidate_field(model_name, field_name)

    remaining_missing = {
        name: f
        for name, f in missing_fields.items()
        if not is_present(name, f) and name not in healed_data
    }

    # Return if all required fields are now satisfied
    if not remaining_missing:
        return healed_data

    skeleton: dict[str, dict[str, Any]] | None = None

    # --------------------------------------------------------------------------
    # Step 2: Tier 1 Zero-LLM Heuristics
    # --------------------------------------------------------------------------
    if remaining_missing and config.enable_heuristics:
        skeleton = extract_json_skeleton(data)
        heuristic_input = {
            name: get_field_type_str(f) for name, f in remaining_missing.items()
        }
        heuristic_matches = find_heuristic_matches(heuristic_input, skeleton)

        for field_name, source_path in heuristic_matches.items():
            found, val = get_by_dot_path(data, source_path)
            if found:
                # Critical Safety Check: verify type compatibility before saving heuristic match
                expected_type = get_field_type_str(remaining_missing[field_name])
                actual_type = skeleton.get(source_path, {}).get(
                    "type",
                    type(val).__name__ if val is not None else "null",
                )
                if is_type_compatible(expected_type, actual_type, val):
                    healed_data[field_name] = val
                    lock_manager.save_mapping(
                        model_name,
                        field_name,
                        source_path,
                        healed_by="heuristic",
                    )
                    healed_notifications[field_name] = (source_path, "heuristic")
                else:
                    logger.warning(
                        "Heuristic suggested '%s' -> '%s' with value %r, which fails type compatibility with '%s'. Discarding.",
                        field_name,
                        source_path,
                        val,
                        expected_type,
                    )

        remaining_missing = {
            name: f
            for name, f in remaining_missing.items()
            if name not in healed_data
        }

    # --------------------------------------------------------------------------
    # Step 3: Tier 2 AI Skeleton Healer
    # --------------------------------------------------------------------------
    if remaining_missing and config.enable_ai:
        if skeleton is None:
            skeleton = extract_json_skeleton(data)

        ai_input = {
            name: {
                "type": get_field_type_str(f),
                "description": str(f.description or ""),
            }
            for name, f in remaining_missing.items()
        }

        ai_matches = heal_with_llm(model_name, ai_input, skeleton, config)
        for field_name, source_path in ai_matches.items():
            found, val = get_by_dot_path(data, source_path)
            if found:
                # Critical Safety Check: verify type compatibility before saving
                expected_type = get_field_type_str(remaining_missing[field_name])
                actual_type = skeleton.get(source_path, {}).get(
                    "type",
                    type(val).__name__ if val is not None else "null",
                )
                if is_type_compatible(expected_type, actual_type, val):
                    healed_data[field_name] = val
                    lock_manager.save_mapping(
                        model_name,
                        field_name,
                        source_path,
                        healed_by="ai",
                    )
                    healed_notifications[field_name] = (source_path, "ai")
                else:
                    logger.warning(
                        "AI suggested '%s' -> '%s' with value %r, which fails type compatibility with '%s'. Discarding.",
                        field_name,
                        source_path,
                        val,
                        expected_type,
                    )

    # Dispatch drift alert notification if any fields were newly healed
    if healed_notifications:
        send_drift_notification(model_name, healed_notifications, config)

    return healed_data


class HealableModel(BaseModel):
    """Drop-in replacement for pydantic.BaseModel with 3-tier self-healing capabilities."""

    heal_config: ClassVar[HealConfig] = HealConfig()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure each subclass gets its own clone of heal_config to avoid shared mutable state
        parent_config = getattr(cls, "heal_config", None)
        if parent_config is not None:
            cls.heal_config = parent_config.clone()
        else:
            cls.heal_config = HealConfig()

    @model_validator(mode="before")
    @classmethod
    def _auto_heal_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        config = getattr(cls, "heal_config", None)
        if config is None:
            config = HealConfig()
        if not config.enabled:
            return data

        return auto_heal_dict(
            data=data,
            model_name=get_model_identifier(cls),
            model_fields=cls.model_fields,
            config=config,
        )


def heal_payload(
    data: Any,
    schema: type[BaseModel],
    config: HealConfig | None = None,
) -> Any:
    """Heals a raw JSON dictionary against a given BaseModel schema."""
    if not isinstance(data, dict):
        return data

    cfg = config or getattr(schema, "heal_config", None) or HealConfig()
    return auto_heal_dict(
        data=data,
        model_name=get_model_identifier(schema),
        model_fields=schema.model_fields,
        config=cfg,
    )


def heal_json(
    schema: type[BaseModel],
    config: HealConfig | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for functions that ingest JSON payloads, automatically healing them before execution.

    Example:
        @heal_json(UserModel)
        def handle_user_webhook(payload: dict[str, Any]):
            user = UserModel.model_validate(payload)
            ...
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if args and isinstance(args[0], dict):
                healed = heal_payload(args[0], schema, config)
                return fn(healed, *args[1:], **kwargs)

            for kw in ("payload", "data", "json_data", "item"):
                if kw in kwargs and isinstance(kwargs[kw], dict):
                    kwargs[kw] = heal_payload(kwargs[kw], schema, config)
                    return fn(*args, **kwargs)

            return fn(*args, **kwargs)

        return wrapper

    return decorator
