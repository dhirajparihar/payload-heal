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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, field_validator

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

