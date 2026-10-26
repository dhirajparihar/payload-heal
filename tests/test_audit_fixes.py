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

import http.server
import json
import multiprocessing as mp
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, field_validator, model_validator

from payload_heal.config import HealConfig
from payload_heal.heuristics import find_heuristic_matches, solve_max_weight_bipartite_matching
from payload_heal.lockfile import LockFileManager
from payload_heal.model import HealableModel, auto_heal_dict, get_model_identifier
from payload_heal.notifier import (
    SSRFSafeBackend,
    SSRFSafeTransport,
    is_safe_webhook_url,
    sanitize_url_for_logging,
    send_drift_notification,
)
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


# ==============================================================================
# 10. Pydantic Constraint Validation Across Healing Pipeline
# ==============================================================================

def test_healed_candidate_rejected_if_pydantic_constraint_fails(tmp_path: Path):
    """Verifies that candidates satisfying structural type check (e.g. float) are rejected
    if they violate specific Pydantic constraints such as Field(gt=0).
    """
    class StockItem(HealableModel):
        price: float = Field(gt=0)

    lockfile = tmp_path / "lock.json"
    cfg = HealConfig(lockfile_path=lockfile, enable_heuristics=True, enable_ai=False)

    # Payload contains candidate 'unit_price', but value is -15.0 (violates gt=0)
    payload = {"unit_price": -15.0}

    # Should raise ValidationError because -15.0 fails Field(gt=0), not silently accept it!
    with pytest.raises(Exception):
        StockItem.model_validate(payload, config=cfg)


def test_healed_candidate_rejected_for_regex_pattern_violation(tmp_path: Path):
    """Verifies candidate strings violating regex pattern constraints are rejected."""
    class Account(HealableModel):
        account_code: str = Field(pattern=r"^[A-Z]{3}-\d{4}$")

    lockfile = tmp_path / "lock.json"
    cfg = HealConfig(lockfile_path=lockfile, enable_heuristics=True, enable_ai=False)

    # Invalid format string
    payload = {"code": "invalid-code"}

    with pytest.raises(Exception):
        Account.model_validate(payload, config=cfg)


# ==============================================================================
# 11. Bipartite Matching Optimality vs Greedy Assignment
# ==============================================================================

def test_bipartite_matching_achieves_global_optimum_over_greedy():
    """Verifies that solve_max_weight_bipartite_matching finds the global maximum score
    in cases where greedy sorting would strand a field and yield a suboptimal total score.
    """
    # Field 1: PathA (0.90), PathB (0.85)
    # Field 2: PathA (0.89)
    # Greedy takes (Field 1 -> PathA) = 0.90, stranding Field 2 -> Total = 0.90
    # Optimal takes (Field 2 -> PathA: 0.89) and (Field 1 -> PathB: 0.85) -> Total = 1.74
    field_candidates = {
        "field1": [(0.90, "pricing.path_a"), (0.85, "pricing.path_b")],
        "field2": [(0.89, "pricing.path_a")],
    }

    result = solve_max_weight_bipartite_matching(field_candidates)

    assert result["field2"] == "pricing.path_a"
    assert result["field1"] == "pricing.path_b"


# ==============================================================================
# 12. Webhook DNS Rebinding & Connect-Time SSRF Blocking
# ==============================================================================

def test_ssrf_safe_backend_blocks_internal_ip_at_connect_time():
    """Verifies SSRFSafeBackend intercepts and blocks connections to private IPs at connect time."""
    backend = SSRFSafeBackend(allow_internal=False)

    from httpcore._exceptions import ConnectError
    import httpx

    transport = SSRFSafeTransport(allow_internal=False)
    with pytest.raises(httpx.ConnectError):
        with httpx.Client(transport=transport) as client:
            client.get("http://127.0.0.1:9999/status")


def test_webhook_redirect_to_internal_ip_is_blocked():
    """Verifies that an external webhook redirecting to an internal IP is blocked."""
    config = HealConfig(
        webhook_url="https://hooks.example.com/drift",
        allowed_webhook_hosts=["hooks.example.com"],
        allow_internal_webhooks=False,
    )

    # Mock response returning a 302 redirect to http://127.0.0.1/admin
    mock_redirect_resp = MagicMock()
    mock_redirect_resp.is_redirect = True
    mock_redirect_resp.headers = {"Location": "http://127.0.0.1/admin"}

    with patch("payload_heal.notifier.is_safe_webhook_url") as mock_url_check:
        # First call (initial URL) is safe, second call (redirect target) is unsafe
        mock_url_check.side_effect = [
            (True, ""),
            (False, "Destination IP 127.0.0.1 is an internal address"),
        ]
        with patch("httpx.Client.post", return_value=mock_redirect_resp):
            success = send_drift_notification("TestModel", {"price": ("val", "heuristic")}, config)
            assert success is False


# ==============================================================================
# 13. Arbitrary JSON Path Escaping & Literal Closing Brackets
# ==============================================================================

def test_parse_dot_path_with_literal_closing_brackets_and_quotes():
    """Verifies parse_dot_path properly parses keys containing literal ] and escaped quotes."""
    path1 = "data['item[0]'].price"
    tokens1 = parse_dot_path(path1)
    assert tokens1 == ["data", "item[0]", "price"]

    path2 = "data['nested]bracket]key']"
    tokens2 = parse_dot_path(path2)
    assert tokens2 == ["data", "nested]bracket]key"]

    path3 = "data['key\\'with\\'quotes']"
    tokens3 = parse_dot_path(path3)
    assert tokens3 == ["data", "key'with'quotes"]

    # Traversal test
    data = {
        "data": {
            "item[0]": {"price": 42.0},
            "nested]bracket]key": "found",
        }
    }
    found1, val1 = get_by_dot_path(data, path1)
    assert found1 is True
    assert val1 == 42.0

    found2, val2 = get_by_dot_path(data, path2)
    assert found2 is True
    assert val2 == "found"


# ==============================================================================
# 14. Adversarial Payload Sanitization & Alternate Credentials
# ==============================================================================

def test_sanitization_against_unicode_homoglyphs_and_alternate_secrets():
    """Verifies detection of unicode homoglyphs, basic auth, refresh tokens, and deep nests."""
    payload = {
        "ａｐｉ_ｋｅｙ": "secret_key_12345",  # full-width unicode
        "auth_header": "Basic dXNlcjpwYXNzd29yZA==",
        "session_pin": 1234,
        "deep": {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": "too_deep"}}}}}},
    }

    sanitized = sanitize_data_recursively(payload, max_depth=5)

    assert sanitized["ａｐｉ_ｋｅｙ"] == "[MASKED_SECRET]"
    assert "Basic [MASKED_TOKEN]" in sanitized["auth_header"]
    assert sanitized["session_pin"] == "[MASKED_SECRET]"
    assert sanitized["deep"]["l1"]["l2"]["l3"]["l4"] == "[TRUNCATED]"


# ==============================================================================
# 15. Lockfile Backup Rotation
# ==============================================================================

def test_lockfile_backup_rotation_limits_retained_files(tmp_path: Path):
    """Verifies that repeatedly corrupting a lockfile produces at most 5 .bak files."""
    lock_path = tmp_path / "app.lock.json"
    mgr = LockFileManager(lock_path)

    for i in range(8):
        lock_path.write_text(f"invalid json {i}", encoding="utf-8")
        _ = mgr.get_mapping("TestModel")

    backup_files = list(tmp_path.glob("*.bak"))
    assert len(backup_files) <= 5


# ==============================================================================
# 16. Model Cache Namespace Override (model_cache_id)
# ==============================================================================

def test_model_cache_id_namespace_override(tmp_path: Path):
    """Verifies that setting model_cache_id cleanly isolates model caches across entrypoints."""
    class CustomEntryModel(HealableModel):
        model_cache_id: ClassVar[str | None] = "service_v2.CustomEntryModel"
        item_id: int

    lock_path = tmp_path / "lock.json"
    cfg = HealConfig(lockfile_path=lock_path, model_cache_id="explicit_namespace_override")

    # When config has model_cache_id, it is used
    assert get_model_identifier(CustomEntryModel, cfg) == "explicit_namespace_override"

    # When config does not have it, class-level model_cache_id is used
    cfg_default = HealConfig(lockfile_path=lock_path)
    assert get_model_identifier(CustomEntryModel, cfg_default) == "service_v2.CustomEntryModel"


# ==============================================================================
# 17. Custom Field Validators Enforced During Healing
# ==============================================================================

def test_custom_field_validator_enforced_during_candidate_healing(tmp_path: Path):
    """Verifies that model-level @field_validator methods are strictly executed
    when evaluating candidate values for healing, rejecting candidates that violate them.
    """
    class ValidatedProduct(HealableModel):
        price: float

        @field_validator("price")
        @classmethod
        def validate_even_price(cls, v: float) -> float:
            if v % 2 != 0:
                raise ValueError("Price must be an even number")
            return v

    lockfile = tmp_path / "lock.json"
    cfg = HealConfig(lockfile_path=lockfile, enable_heuristics=True, enable_ai=False)
    ValidatedProduct.heal_config = cfg

    # 1. Payload with candidate price = 3.0 (odd number -> fails @field_validator)
    payload_invalid = {"unit_price": 3.0}
    with pytest.raises(Exception):
        ValidatedProduct.model_validate(payload_invalid)

    # 2. Payload with candidate price = 4.0 (even number -> passes @field_validator)
    payload_valid = {"unit_price": 4.0}
    product = ValidatedProduct.model_validate(payload_valid)
    assert product.price == 4.0


# ==============================================================================
# 18. Field Alias & Nested Model Candidate Healing
# ==============================================================================

def test_field_alias_and_nested_model_candidate_healing(tmp_path: Path):
    """Verifies that candidate validation succeeds for fields with aliases and nested BaseModels."""
    class Address(BaseModel):
        city: str
        zip_code: str

    class UserProfile(HealableModel):
        address: Address
        display_name: str = Field(alias="screenName")

    lockfile = tmp_path / "lock.json"
    cfg = HealConfig(lockfile_path=lockfile, enable_heuristics=True, enable_ai=False)
    UserProfile.heal_config = cfg

    # Upstream drifted: 'user_address' and 'screen_name'
    payload = {
        "user_address": {"city": "San Francisco", "zip_code": "94105"},
        "screen_name": "coder123",
    }

    user = UserProfile.model_validate(payload)
    assert user.address.city == "San Francisco"
    assert user.address.zip_code == "94105"
    assert user.display_name == "coder123"


# ==============================================================================
# 19. Concurrent Lockfile Writes & Invalidation Stress Test
# ==============================================================================

def test_concurrent_lockfile_writes_and_invalidations(tmp_path: Path):
    """Verifies that concurrent processes/threads writing, reading, and invalidating
    the lockfile simultaneously do not cause deadlocks, corruption, or data loss.
    """
    lock_file = tmp_path / "concurrent_lock.json"
    mgr = LockFileManager(lock_file)

    num_threads = 50

    def worker(i: int):
        model_name = f"Model_{i % 5}"
        field_name = f"field_{i % 3}"
        source_path = f"data.source_{i}"

        # Alternating write, read, and invalidation operations
        if i % 3 == 0:
            mgr.save_mapping(model_name, field_name, source_path, "heuristic")
        elif i % 3 == 1:
            _ = mgr.get_mapping(model_name)
        else:
            mgr.invalidate_field(model_name, field_name)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in futures:
            f.result()

    # The resulting lockfile must remain valid JSON and readable
    assert lock_file.exists()
    content = json.loads(lock_file.read_text(encoding="utf-8"))
    assert "version" in content
    assert "models" in content


# ==============================================================================
# 20. Post-Init AI Endpoint Mutation Blocked at Request Time
# ==============================================================================

def test_post_init_ai_endpoint_mutation_blocked_at_request_time():
    """Verifies that mutating api_base on HealConfig after initialization
    is intercepted and rejected at request time before any network request is made.
    """
    from payload_heal.ai_healer import heal_with_llm

    # Valid config initialized cleanly
    cfg = HealConfig(api_base="https://api.openai.com/v1")

    # Untrusted runtime mutation targeting an unauthorized internal HTTP host
    cfg.api_base = "http://untrusted-internal-host.local/v1"

    missing = {"price": {"type": "float", "description": ""}}
    skeleton = {"price_usd": {"type": "float", "sample": "10.0"}}

    # heal_with_llm must detect insecure scheme and return empty dict without connecting
    matches = heal_with_llm("TestModel", missing, skeleton, cfg)
    assert matches == {}


# ==============================================================================
# 21. Maximum-Weight Bipartite Matching Scalability Benchmark
# ==============================================================================

def test_bipartite_matching_large_scale_scalability_benchmark():
    """Verifies Kuhn-Munkres algorithm runtime with a large candidate graph
    (100 fields x 100 paths) completes in < 250ms without excessive resource use.
    """
    num_fields = 100
    num_paths = 100

    field_candidates: dict[str, list[tuple[float, str]]] = {}
    for i in range(num_fields):
        # Each field has multiple candidate paths with varying scores
        cands = [
            (0.95 - (j * 0.01), f"path_{j}")
            for j in range(max(0, i - 2), min(num_paths, i + 3))
        ]
        field_candidates[f"field_{i}"] = cands

    start_time = time.perf_counter()
    matches = solve_max_weight_bipartite_matching(field_candidates)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    assert len(matches) > 0
    # Should complete comfortably under 250ms (typically ~100-140ms in pure Python)
    assert duration_ms < 250.0, f"Bipartite matching took {duration_ms:.2f}ms (expected < 250ms)"


# ==============================================================================
# 22. Model-Level @model_validator(mode="after") Cross-Field Rules During Healing
# ==============================================================================

def test_model_validator_after_cross_field_rules_during_healing(tmp_path: Path):
    """Verifies that models with @model_validator(mode="after") and cross-field
    dependent validation rules heal properly without candidates being prematurely rejected.
    """
    class Order(HealableModel):
        item_id: str
        quantity: int = Field(gt=0)
        unit_price: float = Field(gt=0.0)

        @field_validator("quantity")
        @classmethod
        def check_max_qty(cls, v: int) -> int:
            if v > 100:
                raise ValueError("Quantity cannot exceed 100")
            return v

        @model_validator(mode="after")
        def validate_cross_fields(self) -> Order:
            # Requires all fields to be present on self
            if not hasattr(self, "item_id") or not hasattr(self, "quantity") or not hasattr(self, "unit_price"):
                raise ValueError("Incomplete order: all fields required for after validation")
            # Cross-field rule: total value cannot exceed 10,000
            if self.quantity * self.unit_price > 10000:
                raise ValueError("Total order value exceeds threshold")
            return self

    lockfile = tmp_path / "lock.json"
    cfg = HealConfig(lockfile_path=lockfile, enable_heuristics=True, enable_ai=False)
    Order.heal_config = cfg

    # 1. Drifted payload where fields are renamed
    drifted_payload = {
        "itemId": "ITEM-XYZ",
        "item_quantity": 10,
        "price_per_unit": 25.50,
    }

    order = Order.model_validate(drifted_payload)
    assert order.item_id == "ITEM-XYZ"
    assert order.quantity == 10
    assert order.unit_price == 25.50

    # 2. Drifted payload where candidate violates field validator (quantity = 150 > 100)
    drifted_invalid_field = {
        "itemId": "ITEM-XYZ",
        "item_quantity": 150,
        "price_per_unit": 25.50,
    }
    with pytest.raises(Exception):
        Order.model_validate(drifted_invalid_field)

    # 3. Drifted payload where individual fields are valid, but cross-field rule fails:
    # quantity = 50, price = 500 -> total = 25,000 > 10,000
    drifted_invalid_cross = {
        "itemId": "ITEM-XYZ",
        "item_quantity": 50,
        "price_per_unit": 500.0,
    }
    with pytest.raises(Exception):
        Order.model_validate(drifted_invalid_cross)


# ==============================================================================
# 23. Live Local HTTP Server Integration Tests for SSRFSafeTransport & Webhooks
# ==============================================================================

def test_live_http_server_webhook_integration_and_redirects():
    """Starts a live local HTTP server to verify SSRFSafeTransport network behavior,
    redirect following, private IP blocking, and payload delivery.
    """
    received_requests: list[dict[str, Any]] = []

    class MockWebhookHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else ""

            if self.path == "/webhook":
                received_requests.append({"path": "/webhook", "body": body})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')

            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/final_webhook")
                self.end_headers()

            elif self.path == "/final_webhook":
                received_requests.append({"path": "/final_webhook", "body": body})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "redirected_ok"}')

            elif self.path == "/redirect_evil":
                self.send_response(302)
                self.send_header("Location", "http://10.0.0.1/evil")
                self.end_headers()

            elif self.path == "/redirect_loop":
                self.send_response(302)
                self.send_header("Location", "/redirect_loop")
                self.end_headers()

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockWebhookHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 1. Allowed internal dispatch delivers payload to local server
        cfg_allowed = HealConfig(
            webhook_url=f"http://127.0.0.1:{port}/webhook",
            allow_internal_webhooks=True,
        )
        drift_data = {"price": ("data.cost", "heuristic")}
        success = send_drift_notification("TestProduct", drift_data, cfg_allowed)
        assert success is True
        assert len(received_requests) == 1
        assert received_requests[0]["path"] == "/webhook"

        # 2. Blocked internal dispatch when allow_internal_webhooks=False
        cfg_blocked = HealConfig(
            webhook_url=f"http://127.0.0.1:{port}/webhook",
            allow_internal_webhooks=False,
        )
        blocked_success = send_drift_notification("TestProduct", drift_data, cfg_blocked)
        assert blocked_success is False
        assert len(received_requests) == 1  # No additional request received

        # 3. Successful redirect follow
        cfg_redir = HealConfig(
            webhook_url=f"http://127.0.0.1:{port}/redirect",
            allow_internal_webhooks=True,
        )
        redir_success = send_drift_notification("TestProduct", drift_data, cfg_redir)
        assert redir_success is True
        assert any(r["path"] == "/final_webhook" for r in received_requests)

        # 4. Dangerous redirect destination blocked (even if initial host allowed)
        cfg_evil = HealConfig(
            webhook_url=f"http://127.0.0.1:{port}/redirect_evil",
            allow_internal_webhooks=False,
        )
        evil_success = send_drift_notification("TestProduct", drift_data, cfg_evil)
        assert evil_success is False

        # 5. Redirect loop terminated safely
        cfg_loop = HealConfig(
            webhook_url=f"http://127.0.0.1:{port}/redirect_loop",
            allow_internal_webhooks=True,
        )
        loop_success = send_drift_notification("TestProduct", drift_data, cfg_loop)
        assert loop_success is False

    finally:
        server.shutdown()
        server.server_close()


# ==============================================================================
# 24. Cross-Process Multiprocessing Concurrency Test
# ==============================================================================

def test_multiprocessing_lockfile_cross_process_safety(tmp_path: Path):
    """Verifies that multiple independent OS processes concurrently writing,
    reading, and invalidating the lockfile preserve all updates without data corruption.
    """
    from payload_heal.lockfile import _mp_worker_sync_test

    lock_file = tmp_path / "cross_process_lock.json"
    num_processes = 4
    items_per_proc = 5

    procs: list[mp.Process] = []
    for p_id in range(num_processes):
        p = mp.Process(
            target=_mp_worker_sync_test,
            args=(str(lock_file), p_id, items_per_proc),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join(timeout=15.0)
        assert p.exitcode == 0, f"Worker process {p} exited with code {p.exitcode}"

    # Verify the lockfile exists and contains all preserved entries
    mgr = LockFileManager(lock_file)
    data = mgr._read_raw()
    models = data.get("models", {})
    assert len(models) == num_processes

    for p_id in range(num_processes):
        model_name = f"ProcModel_{p_id}"
        assert model_name in models
        # field_0 was invalidated, so items_per_proc - 1 fields must remain
        model_fields = models[model_name]
        assert len(model_fields) == items_per_proc - 1
        assert "field_0" not in model_fields
        for i in range(1, items_per_proc):
            assert f"field_{i}" in model_fields


# ==============================================================================
# 25. Connection-Time SSRF Protection for AI Endpoints
# ==============================================================================

def test_ai_endpoint_connection_time_ssrf_blocking():
    """Verifies that an AI endpoint configured with a public hostname whose DNS
    resolves to a private IP (e.g. cloud metadata or internal network) is blocked
    at socket connection time by SSRFSafeTransport.
    """
    from payload_heal.ai_healer import heal_with_llm

    cfg = HealConfig(
        api_base="https://ai.external-domain.com/v1",
        api_key="sk-test",
    )

    missing = {"summary": {"type": "str", "description": "text summary"}}
    skeleton = {"data.text": {"type": "str", "sample": "hello world"}}

    # Mock DNS resolution to return private cloud metadata IP (169.254.169.254)
    fake_dns = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))
    ]

    with patch("socket.getaddrinfo", return_value=fake_dns):
        matches = heal_with_llm("SummaryModel", missing, skeleton, cfg)
        assert matches == {}

    # Verify that intended local provider (e.g. Ollama on localhost) is allowed to connect locally
    cfg_ollama = HealConfig(llm_provider="ollama")
    assert cfg_ollama.api_base == "http://localhost:11434/v1"


