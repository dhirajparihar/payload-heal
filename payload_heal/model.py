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
import threading
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
_HEALING_ACTIVE = threading.local()


def get_model_identifier(schema: type[BaseModel] | str, config: HealConfig | None = None) -> str:
    """Returns a unique, stable identifier for a Pydantic model class.

    Priority:
    1. Explicit class-level attribute `model_cache_id`
    2. Explicit `config.model_cache_id`
    3. Module-qualified class name (e.g. `myapp.models.Product`)
    4. Short class name fallback for __main__
    """
    if isinstance(schema, str):
        return schema

    # 1. Config-level explicit cache identifier (runtime override)
    if config and config.model_cache_id:
        return str(config.model_cache_id)

    # 2. Explicit model-level cache identifier
    explicit_class_id = getattr(schema, "model_cache_id", None)
    if explicit_class_id:
        return str(explicit_class_id)

    mod = getattr(schema, "__module__", "")
    name = getattr(schema, "__name__", "Model")
    if mod and mod not in ("__main__", "builtins"):
        return f"{mod}.{name}"
    return name


_FIELD_SCHEMA_VALIDATOR_CACHE: dict[tuple[type[BaseModel], str], Any] = {}


def _extract_field_core_schema(schema: Any, target_name: str) -> dict[str, Any] | None:
    """Traverses a Pydantic core schema dictionary to extract the specific field's core schema."""
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "model-fields" and "fields" in schema:
        fields = schema["fields"]
        if isinstance(fields, dict) and target_name in fields:
            field_def = fields[target_name]
            if isinstance(field_def, dict) and "schema" in field_def:
                return field_def["schema"]
    for val in schema.values():
        if isinstance(val, dict):
            res = _extract_field_core_schema(val, target_name)
            if res is not None:
                return res
        elif isinstance(val, list):
            for item in val:
                res = _extract_field_core_schema(item, target_name)
                if res is not None:
                    return res
    return None


def validate_field_candidate(
    field_info: FieldInfo,
    field_name: str,
    value: Any,
    model_cls: type[BaseModel] | None = None,
) -> tuple[bool, Any]:
    """Validates a candidate healed value against the field's actual Pydantic schema and constraints.

    Enforces:
    - Base type coercion (e.g., numeric string "49.99" -> float 49.99)
    - Numeric constraints: gt, ge, lt, le, multiple_of
    - String constraints: min_length, max_length, pattern regexes
    - Custom @field_validator methods on model_cls
    - Literal, Enum, Union, and nested model schemas
    - Field aliases with populate_by_name compatibility
    - Preserves normal model semantics: does NOT prematurely trigger whole-model
      @model_validator(mode="after") on incomplete instances with missing sibling fields.

    Returns:
        (is_valid, validated_coerced_value)
    """
    # 1. Native compiled single-field validation using pydantic_core.SchemaValidator
    # This isolates candidate validation strictly to the field itself:
    # - Enforces field type coercion and constraints (gt, min_length, regex, etc.)
    # - Enforces model-level @field_validator(field_name) methods
    # - Validates nested BaseModel, Union, and Enum schemas
    # - Critically, avoids triggering whole-model @model_validator(mode="after") which checks
    #   cross-field rules on incomplete instances where sibling fields are not yet healed.
    if model_cls is not None and hasattr(model_cls, "__pydantic_core_schema__"):
        cache_key = (model_cls, field_name)
        validator = _FIELD_SCHEMA_VALIDATOR_CACHE.get(cache_key)
        if validator is None:
            try:
                from pydantic_core import SchemaValidator

                field_core_schema = _extract_field_core_schema(
                    model_cls.__pydantic_core_schema__, field_name
                )
                if field_core_schema is not None:
                    validator = SchemaValidator(field_core_schema)
                    _FIELD_SCHEMA_VALIDATOR_CACHE[cache_key] = validator
            except Exception:
                validator = None

        if validator is not None:
            try:
                coerced = validator.validate_python(value)
                return True, coerced
            except Exception:
                return False, None

    # 2. Native compiled assignment fallback (for models where core_schema extraction didn't find the field)
    if model_cls is not None and hasattr(model_cls, "__pydantic_validator__"):
        try:
            dummy_instance = model_cls.model_construct()
            validated_instance = model_cls.__pydantic_validator__.validate_assignment(
                dummy_instance, field_name, value
            )
            return True, getattr(validated_instance, field_name)
        except Exception:
            pass

    # 3. Dynamic single-field fallback validator
    try:
        from pydantic import ConfigDict, create_model

        validator_model = create_model(
            "_CandidateValidator",
            __config__=ConfigDict(populate_by_name=True, arbitrary_types_allowed=True),
            **{field_name: (field_info.annotation, field_info)},
        )
        instance = validator_model(**{field_name: value})
        return True, getattr(instance, field_name)
    except Exception:
        return False, None


def get_field_type_str(field_info: FieldInfo) -> str:
    """Extracts a clean, canonical type annotation string from a Pydantic FieldInfo."""
    ann = field_info.annotation
    if ann is None:
        return "Any"
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return "dict"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann).replace("typing.", "")


def auto_heal_dict(
    data: dict[str, Any],
    model_name: str,
    model_fields: dict[str, FieldInfo],
    config: HealConfig,
    model_cls: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Applies the 3-tier healing pipeline to an incoming dictionary payload.

    Tiers:
    0. Fast Path & Lockfile cache lookup (with full Pydantic constraint validation)
    1. Heuristic matcher (with full Pydantic constraint validation)
    2. AI Skeleton Healer (with full Pydantic constraint validation)

    Returns:
        The updated dictionary containing healed field values.
    """
    if not config.enabled:
        return data

    if getattr(_HEALING_ACTIVE, "active", False):
        return data

    _HEALING_ACTIVE.active = True
    try:
        return _auto_heal_dict_impl(
            data,
            model_name,
            model_fields,
            config,
            model_cls=model_cls,
        )
    finally:
        _HEALING_ACTIVE.active = False


def _auto_heal_dict_impl(
    data: dict[str, Any],
    model_name: str,
    model_fields: dict[str, FieldInfo],
    config: HealConfig,
    model_cls: type[BaseModel] | None = None,
) -> dict[str, Any]:
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

    def _apply_healed_field(f_name: str, f_info: FieldInfo, coerced_val: Any) -> None:
        healed_data[f_name] = coerced_val
        alias = getattr(f_info, "alias", None)
        if alias and alias != f_name:
            healed_data[str(alias)] = coerced_val

    # --------------------------------------------------------------------------
    # Step 1: Tier 0 Lockfile Lookup & Re-Drift Invalidation
    # --------------------------------------------------------------------------
    cached_mappings = lock_manager.get_mapping(model_name)
    for field_name in list(missing_fields.keys()):
        if field_name in cached_mappings:
            source_path = cached_mappings[field_name]
            found, val = get_by_dot_path(data, source_path)
            if found:
                field_info = missing_fields[field_name]
                expected_type = get_field_type_str(field_info)
                actual_type = type(val).__name__ if val is not None else "null"

                # Safety Check: Structural compatibility filter + full Pydantic constraint check
                type_ok = is_type_compatible(expected_type, actual_type, val)
                valid, validated_val = (
                    validate_field_candidate(field_info, field_name, val, model_cls=model_cls)
                    if type_ok
                    else (False, None)
                )

                if valid:
                    _apply_healed_field(field_name, field_info, validated_val)
                else:
                    logger.warning(
                        "Cached path '%s' for field '%s' with value %r failed Pydantic field validation. Invalidating cache.",
                        source_path,
                        field_name,
                        val,
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
        aliases = {
            name: str(f.alias)
            for name, f in remaining_missing.items()
            if getattr(f, "alias", None)
        }
        heuristic_matches = find_heuristic_matches(
            heuristic_input,
            skeleton,
            aliases=aliases,
        )

        for field_name, source_path in heuristic_matches.items():
            found, val = get_by_dot_path(data, source_path)
            if found:
                field_info = remaining_missing[field_name]
                expected_type = get_field_type_str(field_info)
                actual_type = skeleton.get(source_path, {}).get(
                    "type",
                    type(val).__name__ if val is not None else "null",
                )

                # Safety Check: Structural filter + full Pydantic constraint check
                type_ok = is_type_compatible(expected_type, actual_type, val)
                valid, validated_val = (
                    validate_field_candidate(field_info, field_name, val, model_cls=model_cls)
                    if type_ok
                    else (False, None)
                )

                if valid:
                    _apply_healed_field(field_name, field_info, validated_val)
                    lock_manager.save_mapping(
                        model_name,
                        field_name,
                        source_path,
                        healed_by="heuristic",
                    )
                    healed_notifications[field_name] = (source_path, "heuristic")
                else:
                    logger.warning(
                        "Heuristic match '%s' -> '%s' with value %r failed Pydantic field validation. Discarding.",
                        field_name,
                        source_path,
                        val,
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
                field_info = remaining_missing[field_name]
                expected_type = get_field_type_str(field_info)
                actual_type = skeleton.get(source_path, {}).get(
                    "type",
                    type(val).__name__ if val is not None else "null",
                )

                # Safety Check: Structural filter + full Pydantic constraint check
                type_ok = is_type_compatible(expected_type, actual_type, val)
                valid, validated_val = (
                    validate_field_candidate(field_info, field_name, val, model_cls=model_cls)
                    if type_ok
                    else (False, None)
                )

                if valid:
                    _apply_healed_field(field_name, field_info, validated_val)
                    lock_manager.save_mapping(
                        model_name,
                        field_name,
                        source_path,
                        healed_by="ai",
                    )
                    healed_notifications[field_name] = (source_path, "ai")
                else:
                    logger.warning(
                        "AI suggested '%s' -> '%s' with value %r failed Pydantic field validation. Discarding.",
                        field_name,
                        source_path,
                        val,
                    )

    # Dispatch drift alert notification if any fields were newly healed
    if healed_notifications:
        send_drift_notification(model_name, healed_notifications, config)

    return healed_data


class HealableModel(BaseModel):
    """Drop-in replacement for pydantic.BaseModel with 3-tier self-healing capabilities."""

    heal_config: ClassVar[HealConfig] = HealConfig()
    model_cache_id: ClassVar[str | None] = None

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
            model_name=get_model_identifier(cls, config),
            model_fields=cls.model_fields,
            config=config,
            model_cls=cls,
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
        model_name=get_model_identifier(schema, cfg),
        model_fields=schema.model_fields,
        config=cfg,
        model_cls=schema if isinstance(schema, type) and issubclass(schema, BaseModel) else None,
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
