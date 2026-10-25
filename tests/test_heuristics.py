"""Tests for Tier-1 Zero-LLM Deterministic Heuristic Matcher."""

from __future__ import annotations

import pytest

from payload_heal.heuristics import (
    find_heuristic_matches,
    is_type_compatible,
    normalize_key,
    score_field_candidate,
    tokenize_key,
)
from payload_heal.skeleton import extract_json_skeleton

# ==============================================================================
# Key Normalization & Tokenization Tests
# ==============================================================================

def test_tokenize_key():
    assert tokenize_key("productPriceUSD") == ["product", "price", "usd"]
    assert tokenize_key("ProductPrice") == ["product", "price"]
    assert tokenize_key("product-price-inr") == ["product", "price", "inr"]
    assert tokenize_key("product_price") == ["product", "price"]
    assert tokenize_key("getHTTPResponseCode") == ["get", "http", "response", "code"]
    assert tokenize_key("") == []


def test_normalize_key():
    assert normalize_key("productPriceUSD") == "product_price_usd"
    assert normalize_key("priceInr") == "price_inr"
    assert normalize_key("price_inr") == "price_inr"
    assert normalize_key("item-title") == "item_title"
    assert normalize_key("ProductPrice") == "product_price"
    assert normalize_key("UserId") == "user_id"


# ==============================================================================
# Type Compatibility Tests
# ==============================================================================

def test_is_type_compatible_primitives():
    # Float accepts float, int, and numeric strings
    assert is_type_compatible("float", "float", 129.99) is True
    assert is_type_compatible("float", "int", 50) is True
    assert is_type_compatible("float", "str", "129.99") is True
    assert is_type_compatible("float", "str", "50") is True
    # Float rejects non-numeric strings
    assert is_type_compatible("float", "str", "MacBook Pro Title") is False
    assert is_type_compatible("float", "str", "") is False

    # Int accepts int and numeric int strings
    assert is_type_compatible("int", "int", 42) is True
    assert is_type_compatible("int", "str", "42") is True
    assert is_type_compatible("int", "str", "not an int") is False

    # Bool accepts bool and boolean-like strings/ints
    assert is_type_compatible("bool", "bool", True) is True
    assert is_type_compatible("bool", "str", "true") is True
    assert is_type_compatible("bool", "str", "false") is True
    assert is_type_compatible("bool", "int", 1) is True
    assert is_type_compatible("bool", "str", "Random String") is False

    # Str accepts str, int, float
    assert is_type_compatible("str", "str", "hello") is True
    assert is_type_compatible("str", "int", 123) is True
    assert is_type_compatible("str", "float", 45.6) is True
    # Str should not coerce dict or list
    assert is_type_compatible("str", "dict", "{}") is False
    assert is_type_compatible("str", "list", "[]") is False


def test_is_type_compatible_unions_and_collections():
    # Optional / Union with None
    assert is_type_compatible("Optional[float]", "float", 10.5) is True
    assert is_type_compatible("Optional[float]", "null", None) is True
    assert is_type_compatible("float | None", "null", None) is True
    assert is_type_compatible("float", "null", None) is False

    # Union[int, str]
    assert is_type_compatible("int | str", "int", 10) is True
    assert is_type_compatible("int | str", "str", "abc") is True

    # Collections
    assert is_type_compatible("list[str]", "list[str]", "['a', 'b']") is True
    assert is_type_compatible("dict[str, Any]", "dict", "{}") is True


# ==============================================================================
# Heuristic Matcher Core Tests
# ==============================================================================

def test_heuristic_exact_leaf_relocation():
    """Requirement: Exact leaf relocation ({"price_usd": 50} moved to {"payload": {"pricing": {"priceUsd": 50}}})."""
    payload = {
        "payload": {
            "pricing": {
                "priceUsd": 50,
            }
        }
    }
    skeleton = extract_json_skeleton(payload)

    missing_fields = {"price_usd": "float"}
    matches = find_heuristic_matches(missing_fields, skeleton)

    assert "price_usd" in matches
    assert matches["price_usd"] == "payload.pricing.priceUsd"


def test_heuristic_type_rejection():
    """Requirement: Type rejection (preventing a string title from matching a float price field even if names are similar)."""
    skeleton = {
        "store.item_price_title": {
            "type": "str",
            "sample": "Special Discount Price Offer",
        },
        "store.product_sku": {
            "type": "str",
            "sample": "SKU-999",
        },
    }

    # Model expects a float price
    missing_fields = {"price": "float"}
    matches = find_heuristic_matches(missing_fields, skeleton)

    # item_price_title has 'price' in its name, but type is incompatible (cannot coerce to float)
    assert matches == {}


def test_heuristic_ambiguity_protection():
    """Requirement: Ambiguity protection (if two nested objects both have "id", do not blindly guess if confidence is tied)."""
    payload = {
        "order": {
            "id": 101,
            "user": {
                "id": 202,
            },
        }
    }
    skeleton = extract_json_skeleton(payload)

    # Missing field is generic "id"
    missing_fields = {"id": "int"}
    matches = find_heuristic_matches(missing_fields, skeleton)

    # Both order.id and order.user.id have leaf name "id" and type int -> tied confidence -> reject
    assert "id" not in matches
    assert matches == {}


def test_heuristic_disambiguation_with_parent_context():
    """When target field has parent context (e.g. user_id), it should match user.id over order.id."""
    payload = {
        "order": {
            "id": 101,
            "user": {
                "id": 202,
            },
        }
    }
    skeleton = extract_json_skeleton(payload)

    # Missing field is specific "user_id"
    missing_fields = {"user_id": "int"}
    matches = find_heuristic_matches(missing_fields, skeleton)

    # Matches order.user.id cleanly because parent 'user' + leaf 'id' == 'user_id'
    assert matches == {"user_id": "order.user.id"}


def test_heuristic_prefix_suffix_and_casing_recovery():
    """Handles casing changes (camelCase to snake_case) and prefix additions (item_title vs title)."""
    payload = {
        "data": {
            "itemTitle": "Ergonomic Mechanical Keyboard",
            "inStock": True,
            "priceInr": 4999.0,
        }
    }
    skeleton = extract_json_skeleton(payload)

    missing_fields = {
        "title": "str",
        "in_stock": "bool",
        "price_inr": "float",
    }
    matches = find_heuristic_matches(missing_fields, skeleton)

    assert matches["title"] == "data.itemTitle"
    assert matches["in_stock"] == "data.inStock"
    assert matches["price_inr"] == "data.priceInr"


def test_heuristic_confidence_threshold_rejection():
    """Unrelated fields should not match if score is below confidence threshold."""
    skeleton = {
        "metadata.created_at": {"type": "str", "sample": "2026-10-25T10:00:00Z"},
    }
    missing_fields = {"shipping_address": "str"}
    matches = find_heuristic_matches(missing_fields, skeleton, confidence_threshold=0.78)
    assert matches == {}


def test_heuristic_empty_inputs():
    assert find_heuristic_matches({}, {"a": {"type": "str", "sample": "val"}}) == {}
    assert find_heuristic_matches({"name": "str"}, {}) == {}


def test_heuristic_competing_fields_conflict_resolution():
    """If two fields both match the same path, the higher confidence match wins."""
    skeleton = {
        "cart.total_price_usd": {"type": "float", "sample": "199.99"},
        "cart.item_count": {"type": "int", "sample": "3"},
    }

    # 'total_price_usd' is an exact match for cart.total_price_usd (score 0.95)
    # 'price' also has token overlap with cart.total_price_usd, but lower score (~0.85)
    missing_fields = {
        "total_price_usd": "float",
        "price": "float",
        "count": "int",
    }
    matches = find_heuristic_matches(missing_fields, skeleton)

    assert matches["total_price_usd"] == "cart.total_price_usd"
    assert matches["count"] == "cart.item_count"
    # 'price' should not steal 'cart.total_price_usd' from 'total_price_usd'
    assert "price" not in matches


def test_heuristic_partial_ambiguity():
    """If one field is ambiguous and another is unambiguous, the unambiguous one must still succeed."""
    skeleton = {
        "order.id": {"type": "int", "sample": "100"},
        "user.id": {"type": "int", "sample": "200"},
        "order.customer_email": {"type": "str", "sample": "customer@example.com"},
    }

    missing_fields = {
        "id": "int",  # ambiguous between order.id and user.id
        "email": "str",  # unambiguous -> matches order.customer_email
    }
    matches = find_heuristic_matches(missing_fields, skeleton)

    assert "id" not in matches
    assert matches["email"] == "order.customer_email"

