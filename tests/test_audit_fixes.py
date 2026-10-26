"""Comprehensive tests verifying fixes for audit findings:
1. Recursive PII and secret masking
2. Cached and heuristic type compatibility verification
3. Global assignment of competing heuristic candidates
4. Model boundary isolation (qualified identifiers)
5. Webhook SSRF protection and safe logging
6. HealConfig mutability isolation across subclasses
7. Lockfile corruption backup and graceful recovery
8. Skipping unnecessary healing for optional fields with defaults
9. Dotted and bracketed JSON key handling
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from payload_heal.config import HealConfig
from payload_heal.heuristics import find_heuristic_matches
from payload_heal.lockfile import LockFileManager
from payload_heal.model import HealableModel, auto_heal_dict, get_model_identifier
from payload_heal.notifier import is_safe_webhook_url, sanitize_url_for_logging, send_drift_notification
from payload_heal.skeleton import (
    extract_json_skeleton,
    format_sample_preview,
    get_by_dot_path,
    mask_pii,
    parse_dot_path,
    sanitize_data_recursively,
)


# ==============================================================================
# 1. Recursive PII and Sensitive Key Sanitization Tests
# ==============================================================================

def test_recursive_pii_and_secret_masking():
    """Verifies that secrets nested in objects/lists and sensitive dictionary keys are masked."""
    nested_payload = {
        "user": {
            "name": "Alice",
            "api_key": "sk-test12345678901234567890",
            "password": "superSecretPassword123!",
            "private_data": {
                "access_token": "ghp_1234567890abcdef1234567890abcdef",
                "aws_key": "AKIAIOSFODNN7EXAMPLE",
                "id_document": "123-45-6789",
                "email": "alice@securecorp.com",
            },
        },
        "credentials": [
            {"token": "secret-token-abcdef123456"},
            {"public_info": "safe_value"},
        ],
        "records": [
            {"token": "secret-token-abcdef123456"},
            {"public_info": "safe_value"},
        ],
    }

    sanitized = sanitize_data_recursively(nested_payload)

    # Sensitive keys replaced with [MASKED_SECRET]
    assert sanitized["user"]["api_key"] == "[MASKED_SECRET]"
    assert sanitized["user"]["password"] == "[MASKED_SECRET]"
    assert sanitized["user"]["private_data"]["access_token"] == "[MASKED_SECRET]"
    assert sanitized["credentials"] == "[MASKED_SECRET]"

    # Pattern based masks
    assert sanitized["user"]["private_data"]["aws_key"] == "[MASKED_AWS_KEY]"
    assert sanitized["user"]["private_data"]["id_document"] == "[MASKED_SSN]"
    assert sanitized["user"]["private_data"]["email"] == "[MASKED_EMAIL]"

    # Nested list dictionaries
    assert sanitized["records"][0]["token"] == "[MASKED_SECRET]"
    assert sanitized["records"][1]["public_info"] == "safe_value"
    assert sanitized["user"]["name"] == "Alice"


def test_format_sample_preview_recursively_masks_secrets():
    """Verifies format_sample_preview does not leak nested secrets or tokens."""
    sample = {
        "admin": {
            "password_hash": "my_secret_hash_value",
            "notes": "Contact admin at security@example.com",
        }
    }
    preview = format_sample_preview(sample, max_len=150)
    assert "my_secret_hash_value" not in preview
    assert "[MASKED_SECRET]" in preview
    assert "security@example.com" not in preview
    assert "[MASKED_EMAIL]" in preview


# ==============================================================================
# 2. Type Checking Across All Tiers (Cached & Heuristics)
# ==============================================================================

def test_cached_mapping_rejected_and_invalidated_on_type_incompatibility(tmp_path: Path):
    """Verifies Tier 0 rejects cached mapping if value type is incompatible with model field."""
    lockfile_path = tmp_path / "lock.json"
    lock_mgr = LockFileManager(lockfile_path)

    # Save a mapping where 'price' was previously mapped to 'meta.title' (a string)
    lock_mgr.save_mapping("ProductModel", "price", "meta.title", "heuristic")

    config = HealConfig(lockfile_path=lockfile_path, enable_heuristics=False, enable_ai=False)

    class ProductModel(BaseModel):
        price: float

    # Incoming payload has meta.title = "NonNumericBookTitle" (not coercible to float)
    payload = {"meta": {"title": "The Hitchhiker's Guide"}}

    healed = auto_heal_dict(
        data=payload,
        model_name="ProductModel",
        model_fields=ProductModel.model_fields,
        config=config,
    )

    # Price should NOT be accepted from cache because it's incompatible!
    assert "price" not in healed

    # Stale cache mapping should have been invalidated
    assert lock_mgr.get_mapping("ProductModel") == {}


def test_heuristic_match_rejected_if_type_incompatible(tmp_path: Path):
    """Verifies Tier 1 heuristics rejects candidate if sample value cannot coerce."""
    lockfile_path = tmp_path / "lock.json"
    config = HealConfig(lockfile_path=lockfile_path, enable_heuristics=True, enable_ai=False)

    class Item(BaseModel):
        count: int

    # Payload has 'item_count' but the value is non-numeric text "unlimited"
    payload = {"item_count": "unlimited"}

    healed = auto_heal_dict(
        data=payload,
        model_name="Item",
        model_fields=Item.model_fields,
        config=config,
    )

    assert "count" not in healed


# ==============================================================================
# 3. Global Assignment of Competing Heuristic Candidates
# ==============================================================================

def test_competing_heuristic_candidates_global_assignment():
    """When Field A and Field B compete for overlapping candidate paths,
    the matcher should resolve globally so that both fields find matches
    when valid second-best alternatives exist.
    """
    missing_fields = {
        "price": "float",
        "item_price": "float",
    }
    skeleton = {
        "pricing.price": {"type": "float", "sample": "29.99"},
        "pricing.item_price": {"type": "float", "sample": "29.99"},
    }

    matches = find_heuristic_matches(missing_fields, skeleton)

    # Both fields should be assigned to distinct paths
    assert len(matches) == 2
    assert "price" in matches
    assert "item_price" in matches
    assert matches["price"] != matches["item_price"]


# ==============================================================================
# 4. Model Boundary Isolation (Qualified Identifiers)
# ==============================================================================

def test_model_boundary_isolation_prevents_cache_collision(tmp_path: Path):
    """Verifies that two models with the same class name in different modules/scopes
    do not overwrite each other's lockfile entries.
    """
    lockfile_path = tmp_path / "lock.json"
    lock_mgr = LockFileManager(lockfile_path)

    id_module_a = "sales.models.Product"
    id_module_b = "inventory.models.Product"

    lock_mgr.save_mapping(id_module_a, "price", "sale_price", "heuristic")
    lock_mgr.save_mapping(id_module_b, "price", "unit_cost", "heuristic")

    # Mappings are strictly isolated by qualified identifier
    assert lock_mgr.get_mapping(id_module_a) == {"price": "sale_price"}
    assert lock_mgr.get_mapping(id_module_b) == {"price": "unit_cost"}


# ==============================================================================
# 5. Webhook SSRF Protection and Safe Logging
# ==============================================================================

def test_webhook_url_validation_blocks_ssrf():
    """Verifies webhook validation blocks private IP ranges and local addresses."""
    # Private / loopback addresses blocked by default
    assert not is_safe_webhook_url("http://127.0.0.1:8000/webhook")[0]
    assert not is_safe_webhook_url("http://localhost:5000/hook")[0]
    assert not is_safe_webhook_url("http://169.254.169.254/latest/meta-data")[0]
    assert not is_safe_webhook_url("http://10.0.0.1/notify")[0]
    assert not is_safe_webhook_url("http://192.168.1.1/webhook")[0]

    # Non-HTTP schemes blocked
    assert not is_safe_webhook_url("ftp://example.com/webhook")[0]
    assert not is_safe_webhook_url("file:///etc/passwd")[0]

    # Valid external HTTPS URL allowed
    assert is_safe_webhook_url("https://hooks.slack.com/services/T00/B00/X00")[0]


def test_webhook_host_allowlist_enforced():
    """Verifies host allowlist restricts webhooks to approved destinations."""
    allowed = ["hooks.slack.com", "discord.com"]

    assert is_safe_webhook_url("https://hooks.slack.com/services/test", allowed_hosts=allowed)[0]
    assert is_safe_webhook_url("https://discord.com/api/webhooks/test", allowed_hosts=allowed)[0]
    assert not is_safe_webhook_url("https://evil-attacker.com/leak", allowed_hosts=allowed)[0]


def test_sanitize_url_for_logging():
    """Verifies credentials, tokens, and query params are redacted from webhook logs."""
    url = "https://user:password123@hooks.slack.com/services/T123/B456/SECRET_TOKEN?api_key=SECRET"
    sanitized = sanitize_url_for_logging(url)

    assert "password123" not in sanitized
    assert "SECRET_TOKEN" not in sanitized
    assert "api_key=SECRET" not in sanitized
    assert "***" in sanitized


# ==============================================================================
# 6. HealConfig Mutability Isolation
# ==============================================================================

def test_heal_config_isolation_across_subclasses():
    """Modifying heal_config on one subclass must not mutate other models or the base class."""
    class ModelA(HealableModel):
        x: int

    class ModelB(HealableModel):
        y: int

    # Mutate ModelA config
    ModelA.heal_config.enabled = False

    # ModelB and base HealableModel must remain enabled
    assert ModelA.heal_config.enabled is False
    assert ModelB.heal_config.enabled is True
    assert HealableModel.heal_config.enabled is True


# ==============================================================================
# 7. Lockfile Corruption Recovery
# ==============================================================================

def test_corrupted_lockfile_backed_up_and_recovered(tmp_path: Path):
    """Verifies a corrupted JSON lockfile is preserved as a .bak file and gracefully reset."""
    lockfile_path = tmp_path / ".payload_heal_lock.json"
    lockfile_path.write_text("{ this is invalid json !!!", encoding="utf-8")

    mgr = LockFileManager(lockfile_path)
    mapping = mgr.get_mapping("TestModel")

    # Gracefully returns empty mapping instead of crashing
    assert mapping == {}

    # Backup file should have been created
    backup_files = list(tmp_path.glob("*.bak"))
    assert len(backup_files) == 1
    assert backup_files[0].read_text(encoding="utf-8") == "{ this is invalid json !!!"


# ==============================================================================
# 8. Optional Fields with Defaults Skip Unnecessary Healing
# ==============================================================================

def test_optional_fields_with_defaults_do_not_trigger_healing():
    """Optional fields with defaults should not enter the missing fields set."""
    class UserModel(BaseModel):
        username: str
        bio: Optional[str] = "Default bio"
        score: int = 0

    # Payload has username, but bio and score are omitted
    payload = {"username": "johndoe"}

    config = HealConfig(enable_heuristics=True, enable_ai=False)

    # auto_heal_dict should immediately return fast-path because all required fields are satisfied
    healed = auto_heal_dict(
        data=payload,
        model_name="UserModel",
        model_fields=UserModel.model_fields,
        config=config,
    )

    # No healing needed, fast path returned unchanged payload
    assert healed == payload


# ==============================================================================
# 9. Dotted and Bracketed Key Extraction & Traversal
# ==============================================================================

def test_dotted_and_bracketed_keys_extraction_and_traversal():
    """Payloads with literal dots in keys should be safely bracketed and traversed."""
    payload = {
        "price.usd": 99.99,
        "nested": {
            "rate.eur": 85.50,
            "items[0]": "special",
        },
    }

    skeleton = extract_json_skeleton(payload)

    # Keys containing dots should be wrapped in brackets
    assert "['price.usd']" in skeleton
    assert "nested['rate.eur']" in skeleton

    # Traversal with parse_dot_path and get_by_dot_path
    found1, val1 = get_by_dot_path(payload, "['price.usd']")
    assert found1 is True
    assert val1 == 99.99

    found2, val2 = get_by_dot_path(payload, "nested['rate.eur']")
    assert found2 is True
    assert val2 == 85.50
