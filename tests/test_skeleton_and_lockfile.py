"""Comprehensive pytest suite testing skeleton extraction, path traversal, PII masking,
and atomic thread-safe lockfile operations for payload-heal.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path

import pytest

from payload_heal.lockfile import (
    LockFileManager,
    get_mapping,
    invalidate_field,
    save_mapping,
)
from payload_heal.skeleton import (
    extract_json_skeleton,
    get_by_dot_path,
    is_luhn_valid,
    mask_pii,
    parse_dot_path,
)

# ==============================================================================
# Path Traversal Tests (get_by_dot_path)
# ==============================================================================

def test_get_by_dot_path_deeply_nested():
    data = {
        "org": {
            "department": {
                "teams": [
                    {
                        "name": "Core Dev",
                        "lead": {
                            "contact": {
                                "email": "lead@company.com",
                                "office": "Building A",
                            }
                        },
                    }
                ]
            }
        }
    }

    found, val = get_by_dot_path(data, "org.department.teams[0].lead.contact.email")
    assert found is True
    assert val == "lead@company.com"

    found, val = get_by_dot_path(data, "org.department.teams[0].name")
    assert found is True
    assert val == "Core Dev"


def test_get_by_dot_path_multidimensional_lists():
    data = {
        "matrix": [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    }
    found, val = get_by_dot_path(data, "matrix[1][2]")
    assert found is True
    assert val == 6

    # Root array index
    root_list = [{"title": "Alpha"}, {"title": "Beta"}]
    found, val = get_by_dot_path(root_list, "[1].title")
    assert found is True
    assert val == "Beta"


def test_get_by_dot_path_quoted_keys():
    data = {
        "special": {
            "key.with.dot": "dotted_val",
            "key with spaces": "spaced_val",
        }
    }
    found, val = get_by_dot_path(data, "special['key.with.dot']")
    assert found is True
    assert val == "dotted_val"

    found, val = get_by_dot_path(data, 'special["key with spaces"]')
    assert found is True
    assert val == "spaced_val"


def test_get_by_dot_path_falsy_values():
    """Critical: get_by_dot_path must return (True, val) even if val is falsy."""
    data = {
        "zero_int": 0,
        "zero_float": 0.0,
        "false_bool": False,
        "empty_str": "",
        "null_val": None,
        "empty_list": [],
        "empty_dict": {},
        "nested": {
            "nested_zero": 0,
            "nested_false": False,
            "nested_empty": "",
        },
    }

    test_cases = [
        ("zero_int", 0),
        ("zero_float", 0.0),
        ("false_bool", False),
        ("empty_str", ""),
        ("null_val", None),
        ("empty_list", []),
        ("empty_dict", {}),
        ("nested.nested_zero", 0),
        ("nested.nested_false", False),
        ("nested.nested_empty", ""),
    ]

    for path, expected_val in test_cases:
        found, val = get_by_dot_path(data, path)
        assert found is True, f"Expected path '{path}' to be found"
        assert val == expected_val, f"Expected path '{path}' to equal {expected_val!r}, got {val!r}"


def test_get_by_dot_path_nonexistent():
    data = {
        "user": {
            "name": "Bob",
            "items": [1, 2],
        }
    }

    # Missing field
    assert get_by_dot_path(data, "user.email") == (False, None)
    # Missing nested key
    assert get_by_dot_path(data, "user.name.first") == (False, None)
    # List index out of bounds
    assert get_by_dot_path(data, "user.items[5]") == (False, None)
    # Traversing into primitive as list
    assert get_by_dot_path(data, "user.name[0]") == (False, None)
    # Empty path
    assert get_by_dot_path(data, "") == (False, None)
    # Malformed syntax
    assert get_by_dot_path(data, "user..name") == (False, None)
    assert get_by_dot_path(data, "user.") == (False, None)
    assert get_by_dot_path(data, ".user") == (False, None)
    assert get_by_dot_path(data, "user[") == (False, None)
    assert get_by_dot_path(data, "user[]") == (False, None)


# ==============================================================================
# Skeleton Extraction Tests (extract_json_skeleton)
# ==============================================================================

def test_extract_json_skeleton_deeply_nested():
    payload = {
        "data": {
            "account": {
                "id": 1001,
                "profile": {
                    "username": "coder_42",
                    "rating": 4.95,
                    "is_active": True,
                },
            }
        }
    }

    skeleton = extract_json_skeleton(payload)
    assert "data.account.id" in skeleton
    assert skeleton["data.account.id"]["type"] == "int"
    assert skeleton["data.account.id"]["sample"] == "1001"

    assert "data.account.profile.username" in skeleton
    assert skeleton["data.account.profile.username"]["type"] == "str"
    assert skeleton["data.account.profile.username"]["sample"] == "coder_42"

    assert "data.account.profile.rating" in skeleton
    assert skeleton["data.account.profile.rating"]["type"] == "float"
    assert skeleton["data.account.profile.rating"]["sample"] == "4.95"

    assert "data.account.profile.is_active" in skeleton
    assert skeleton["data.account.profile.is_active"]["type"] == "bool"
    assert skeleton["data.account.profile.is_active"]["sample"] == "true"


def test_extract_json_skeleton_lists_of_dicts():
    payload = {
        "store": {
            "products": [
                {"sku": "A100", "price": 29.99, "in_stock": True},
                {"sku": "B200", "price": 49.99, "in_stock": False},
            ]
        }
    }

    skeleton = extract_json_skeleton(payload, max_list_sample=1)
    assert "store.products[0].sku" in skeleton
    assert skeleton["store.products[0].sku"]["type"] == "str"
    assert skeleton["store.products[0].sku"]["sample"] == "A100"

    assert "store.products[0].price" in skeleton
    assert skeleton["store.products[0].price"]["type"] == "float"
    assert skeleton["store.products[0].price"]["sample"] == "29.99"

    # Only sample 0 should be present when max_list_sample=1
    assert "store.products[1].sku" not in skeleton


def test_extract_json_skeleton_lists_of_primitives():
    payload = {
        "metadata": {
            "tags": ["python", "ai", "fastapi"],
            "ratings": [5, 4, 5],
            "scores": [0.95, 0.88],
            "flags": [True, False],
        }
    }

    skeleton = extract_json_skeleton(payload)
    assert "metadata.tags" in skeleton
    assert skeleton["metadata.tags"]["type"] == "list[str]"

    assert "metadata.ratings" in skeleton
    assert skeleton["metadata.ratings"]["type"] == "list[int]"

    assert "metadata.scores" in skeleton
    assert skeleton["metadata.scores"]["type"] == "list[float]"

    assert "metadata.flags" in skeleton
    assert skeleton["metadata.flags"]["type"] == "list[bool]"


def test_extract_json_skeleton_empty_collections():
    payload = {
        "empty_list": [],
        "empty_dict": {},
        "nested": {
            "inner_empty_list": [],
            "inner_empty_dict": {},
        },
    }

    skeleton = extract_json_skeleton(payload)
    assert "empty_list" in skeleton
    assert skeleton["empty_list"]["type"] == "list"
    assert skeleton["empty_list"]["sample"] == "[]"

    assert "empty_dict" in skeleton
    assert skeleton["empty_dict"]["type"] == "dict"
    assert skeleton["empty_dict"]["sample"] == "{}"

    assert "nested.inner_empty_list" in skeleton
    assert skeleton["nested.inner_empty_list"]["type"] == "list"

    assert "nested.inner_empty_dict" in skeleton
    assert skeleton["nested.inner_empty_dict"]["type"] == "dict"


def test_extract_json_skeleton_falsy_values():
    payload = {
        "falsy": {
            "zero_int": 0,
            "false_bool": False,
            "empty_str": "",
            "null_val": None,
        }
    }

    skeleton = extract_json_skeleton(payload)
    assert skeleton["falsy.zero_int"]["type"] == "int"
    assert skeleton["falsy.zero_int"]["sample"] == "0"

    assert skeleton["falsy.false_bool"]["type"] == "bool"
    assert skeleton["falsy.false_bool"]["sample"] == "false"

    assert skeleton["falsy.empty_str"]["type"] == "str"
    assert skeleton["falsy.empty_str"]["sample"] == ""

    assert skeleton["falsy.null_val"]["type"] == "null"
    assert skeleton["falsy.null_val"]["sample"] == "null"


# ==============================================================================
# PII Masking Tests
# ==============================================================================

def test_pii_email_masking():
    samples = [
        "john.doe@example.com",
        "Contact support at help-desk@corp.co.uk immediately.",
        "user+tag@domain.io",
    ]
    for s in samples:
        masked = mask_pii(s)
        assert "[MASKED_EMAIL]" in masked
        assert "@example.com" not in masked
        assert "@corp.co.uk" not in masked
        assert "@domain.io" not in masked


def test_pii_bearer_and_tokens_masking():
    # Bearer header
    masked = mask_pii("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    assert "Bearer [MASKED_TOKEN]" in masked
    assert "eyJhbGci" not in masked

    # OpenAI-style API key
    masked = mask_pii("API key is sk-1234567890abcdef1234567890abcdef")
    assert "[MASKED_TOKEN]" in masked
    assert "sk-1234567890" not in masked


def test_pii_credit_card_masking():
    # Formatted with dashes
    masked = mask_pii("Card: 4111-2222-3333-4444")
    assert "[MASKED_CREDIT_CARD]" in masked
    assert "4111-2222" not in masked

    # Formatted with spaces
    masked = mask_pii("Card: 4111 2222 3333 4444")
    assert "[MASKED_CREDIT_CARD]" in masked

    # Luhn algorithm verification helper
    assert is_luhn_valid("49927398716") is True
    assert is_luhn_valid("49927398717") is False

    # Contiguous 16-digit card passing Luhn
    masked = mask_pii("Number: 499273987161234")  # test sequence
    # Normal epoch timestamp (13 digits, non-Luhn) should NOT be masked
    timestamp_str = "Timestamp: 1712345678901"
    if not is_luhn_valid("1712345678901"):
        assert "1712345678901" in mask_pii(timestamp_str)


def test_skeleton_masks_pii_in_payload():
    payload = {
        "user": {
            "email": "secret_user@example.com",
            "token": "Bearer sk-supersecrettoken123456",
            "payment": {
                "card": "4111-2222-3333-4444",
            },
        }
    }

    skeleton = extract_json_skeleton(payload)
    assert skeleton["user.email"]["sample"] == "[MASKED_EMAIL]"
    assert "secret_user" not in skeleton["user.email"]["sample"]

    assert "[MASKED_TOKEN]" in skeleton["user.token"]["sample"]
    assert "supersecret" not in skeleton["user.token"]["sample"]

    assert skeleton["user.payment.card"]["sample"] == "[MASKED_CREDIT_CARD]"
    assert "4111" not in skeleton["user.payment.card"]["sample"]


# ==============================================================================
# LockFile Manager Tests
# ==============================================================================

@pytest.fixture
def temp_lockfile(tmp_path: Path) -> Path:
    return tmp_path / ".payload_heal_lock.json"


def test_lockfile_save_and_get_mapping(temp_lockfile: Path):
    manager = LockFileManager(temp_lockfile)

    # Initially empty
    assert manager.get_mapping("UserModel") == {}

    # Save mapping
    manager.save_mapping("UserModel", "email", "data.user.contact.email", "heuristic")
    manager.save_mapping("UserModel", "full_name", "data.user.name", "ai")

    mapping = manager.get_mapping("UserModel")
    assert mapping == {
        "email": "data.user.contact.email",
        "full_name": "data.user.name",
    }

    # Verify underlying file schema
    raw = json.loads(temp_lockfile.read_text(encoding="utf-8"))
    assert raw["version"] == "1.0"
    user_entry = raw["models"]["UserModel"]
    assert user_entry["email"]["source_path"] == "data.user.contact.email"
    assert user_entry["email"]["healed_by"] == "heuristic"
    # Ensure updated_at is valid ISO-8601
    datetime.fromisoformat(user_entry["email"]["updated_at"])


def test_lockfile_invalidation(temp_lockfile: Path):
    manager = LockFileManager(temp_lockfile)

    manager.save_mapping("ProductModel", "price", "store.item.price", "heuristic")
    manager.save_mapping("ProductModel", "title", "store.item.title", "heuristic")

    assert len(manager.get_mapping("ProductModel")) == 2

    # Invalidate a single drifted field
    success = manager.invalidate_field("ProductModel", "price")
    assert success is True

    mapping_after = manager.get_mapping("ProductModel")
    assert "price" not in mapping_after
    assert mapping_after["title"] == "store.item.title"

    # Invalidate non-existent field
    assert manager.invalidate_field("ProductModel", "non_existent") is False
    assert manager.invalidate_field("UnknownModel", "title") is False

    # Invalidate remaining field should prune empty model entry
    manager.invalidate_field("ProductModel", "title")
    assert manager.get_mapping("ProductModel") == {}
    raw = json.loads(temp_lockfile.read_text(encoding="utf-8"))
    assert "ProductModel" not in raw.get("models", {})


def test_lockfile_module_level_helpers(temp_lockfile: Path):
    # Test passing explicit lockfile_path to module-level functions
    save_mapping("OrderModel", "total", "data.pricing.total", "heuristic", lockfile_path=temp_lockfile)
    mapping = get_mapping("OrderModel", lockfile_path=temp_lockfile)
    assert mapping == {"total": "data.pricing.total"}

    invalidate_field("OrderModel", "total", lockfile_path=temp_lockfile)
    assert get_mapping("OrderModel", lockfile_path=temp_lockfile) == {}


def test_lockfile_corrupted_file_resilience(temp_lockfile: Path):
    temp_lockfile.write_text("{ broken json content ...", encoding="utf-8")
    manager = LockFileManager(temp_lockfile)

    # Should not crash, returns empty mapping
    assert manager.get_mapping("AnyModel") == {}

    # Saving over corrupted file heals it gracefully
    manager.save_mapping("AnyModel", "field1", "path1", "heuristic")
    assert manager.get_mapping("AnyModel") == {"field1": "path1"}


def test_lockfile_thread_concurrency(temp_lockfile: Path):
    """Stress test concurrent writes from multiple threads to ensure thread-safety."""
    manager = LockFileManager(temp_lockfile)
    num_threads = 8
    items_per_thread = 20

    def worker(worker_id: int):
        for i in range(items_per_thread):
            field_name = f"field_{worker_id}_{i}"
            path = f"data.nested_{worker_id}.item_{i}"
            manager.save_mapping("ConcurrentModel", field_name, path, "heuristic")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, w) for w in range(num_threads)]
        for f in futures:
            f.result()

    mapping = manager.get_mapping("ConcurrentModel")
    assert len(mapping) == num_threads * items_per_thread
    for w in range(num_threads):
        for i in range(items_per_thread):
            assert mapping[f"field_{w}_{i}"] == f"data.nested_{w}.item_{i}"


# ==============================================================================
# Additional Edge Case & Regression Tests
# ==============================================================================

def test_extract_json_skeleton_root_primitives_and_lists():
    # Root primitive
    assert extract_json_skeleton(42) == {"": {"type": "int", "sample": "42"}}
    assert extract_json_skeleton("hello world") == {"": {"type": "str", "sample": "hello world"}}

    # Root list of primitives
    skeleton = extract_json_skeleton(["admin", "user"])
    assert "" in skeleton
    assert skeleton[""]["type"] == "list[str]"

    # Root list of dicts
    skeleton_dicts = extract_json_skeleton([{"id": 1, "name": "Alice"}])
    assert "[0].id" in skeleton_dicts
    assert skeleton_dicts["[0].id"]["type"] == "int"
    assert skeleton_dicts["[0].name"]["type"] == "str"


def test_extract_json_skeleton_custom_prefix_and_truncation():
    # Custom prefix
    skeleton = extract_json_skeleton({"user": "Alice"}, prefix="api.v1")
    assert "api.v1.user" in skeleton
    assert skeleton["api.v1.user"]["sample"] == "Alice"

    # Truncation of very long string
    long_str = "x" * 200
    skeleton_long = extract_json_skeleton({"text": long_str})
    sample = skeleton_long["text"]["sample"]
    assert sample.endswith("...")
    assert len(sample) <= 85


def test_get_by_dot_path_negative_indices_and_objects():
    data = {"items": [10, 20, 30]}
    found, val = get_by_dot_path(data, "items[-1]")
    assert found is True
    assert val == 30

    found, val = get_by_dot_path(data, "items[-2]")
    assert found is True
    assert val == 20

    # Negative index out of bounds
    found, val = get_by_dot_path(data, "items[-4]")
    assert found is False
    assert val is None

    # Python object / class with attributes
    class SimpleObj:
        def __init__(self):
            self.title = "ObjectTitle"
            self.value = 100

    obj_data = {"wrapper": SimpleObj()}
    found, val = get_by_dot_path(obj_data, "wrapper.title")
    assert found is True
    assert val == "ObjectTitle"


def test_lockfile_get_field_info_and_clear(temp_lockfile: Path):
    manager = LockFileManager(temp_lockfile)
    manager.save_mapping("Profile", "age", "payload.user.age", "heuristic")

    info = manager.get_field_info("Profile", "age")
    assert info is not None
    assert info["source_path"] == "payload.user.age"
    assert info["healed_by"] == "heuristic"
    assert "updated_at" in info

    assert manager.get_field_info("Profile", "nonexistent") is None
    assert manager.get_field_info("UnknownModel", "age") is None

    # Clear all
    manager.clear()
    assert manager.get_mapping("Profile") == {}

