"""End-to-end integration tests for HealableModel and the 3-tier self-healing architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field

from payload_heal import HealConfig, HealableModel, heal_json
from payload_heal.lockfile import LockFileManager

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_lockfile(tmp_path: Path) -> Path:
    return tmp_path / ".payload_heal_lock.json"


# ==============================================================================
# Test 1: Normal Payload Fast Path (0ms Overhead)
# ==============================================================================

def test_normal_payload_takes_fast_path(temp_lockfile: Path):
    """Test 1: Normal payload takes the 0ms fast path without reading lockfile or skeletonizer."""

    class Product(HealableModel):
        heal_config: ClassVar[HealConfig] = HealConfig(lockfile_path=temp_lockfile)
        name: str
        price: float

    payload = {"name": "Mechanical Keyboard", "price": 99.99}

    # Spy on LockFileManager.get_mapping and extract_json_skeleton
    with patch("payload_heal.model.LockFileManager.get_mapping") as mock_lock, \
         patch("payload_heal.model.extract_json_skeleton") as mock_skeleton:
        product = Product.model_validate(payload)

        # Neither lockfile nor skeletonizer should ever be touched on fast path
        mock_lock.assert_not_called()
        mock_skeleton.assert_not_called()

    assert product.name == "Mechanical Keyboard"
    assert product.price == 99.99


# ==============================================================================
# Test 2: Drifted Payload Heals via Tier 1 Heuristics
# ==============================================================================

def test_drifted_payload_heals_via_tier1_heuristics(temp_lockfile: Path):
    """Test 2: Drifted payload heals via Tier 1 heuristics and writes to lockfile."""

    class Product(HealableModel):
        heal_config: ClassVar[HealConfig] = HealConfig(lockfile_path=temp_lockfile)
        product_name: str
        price_usd: float

    drifted_payload = {
        "status": "success",
        "data": {
            "item": {
                "productName": "Ergonomic Standing Desk",
                "priceUsd": 350.0,
            }
        },
    }

    product = Product.model_validate(drifted_payload)

    assert product.product_name == "Ergonomic Standing Desk"
    assert product.price_usd == 350.0

    # Verify mappings persisted to lockfile with healed_by="heuristic"
    lock_mgr = LockFileManager(temp_lockfile)
    mapping = lock_mgr.get_mapping("Product")
    assert mapping["product_name"] == "data.item.productName"
    assert mapping["price_usd"] == "data.item.priceUsd"

    field_info = lock_mgr.get_field_info("Product", "product_name")
    assert field_info["healed_by"] == "heuristic"


# ==============================================================================
# Test 3 & 4: Radically Renamed Payload via Tier 2 AI & Tier 0 Cache
# ==============================================================================

def test_radically_renamed_payload_heals_via_ai_and_caches_to_tier0(temp_lockfile: Path):
    """Test 3: Radically renamed payload ({"vehicle_data": {"amt_inr": "450000"}} -> price: float)
    heals via mocked Tier 2 AI, coerces string "450000" to 450000.0, and writes to lockfile.

    Test 4: Second call with the same radically renamed payload heals via Tier 0 lockfile
    with ZERO calls to the AI healer (mock_ai.assert_not_called()).
    """

    class Vehicle(HealableModel):
        heal_config: ClassVar[HealConfig] = HealConfig(lockfile_path=temp_lockfile)
        price: float = Field(description="Total price of the vehicle")

    radical_payload = {
        "vehicle_data": {
            "amt_inr": "450000",
        }
    }

    # Step A (Test 3): First call triggers Tier 2 AI Healer
    with patch("payload_heal.model.heal_with_llm") as mock_ai:
        mock_ai.return_value = {"price": "vehicle_data.amt_inr"}

        v1 = Vehicle.model_validate(radical_payload)
        mock_ai.assert_called_once()

    # Verified: Pydantic coerced string "450000" to float 450000.0
    assert v1.price == 450000.0
    assert isinstance(v1.price, float)

    # Verify lockfile cached the mapping under "ai"
    lock_mgr = LockFileManager(temp_lockfile)
    assert lock_mgr.get_mapping("Vehicle") == {"price": "vehicle_data.amt_inr"}
    info = lock_mgr.get_field_info("Vehicle", "price")
    assert info["healed_by"] == "ai"

    # Step B (Test 4): Second call with same payload uses Tier 0 Cache
    with patch("payload_heal.model.heal_with_llm") as mock_ai_2:
        v2 = Vehicle.model_validate(radical_payload)
        # AI Healer must NOT be called on subsequent invocations
        mock_ai_2.assert_not_called()

    assert v2.price == 450000.0


# ==============================================================================
# Test 5: Re-Drift Invalidation & Re-Healing
# ==============================================================================

def test_api_changes_second_time_redrift_invalidation(temp_lockfile: Path):
    """Test 5: API changes a second time (Re-Drift) -> invalidates stale lockfile entry and re-heals."""

    class Vehicle(HealableModel):
        heal_config: ClassVar[HealConfig] = HealConfig(lockfile_path=temp_lockfile)
        price: float

    # Pre-populate lockfile with previous drift
    lock_mgr = LockFileManager(temp_lockfile)
    lock_mgr.save_mapping("Vehicle", "price", "vehicle_data.amt_inr", "ai")
    assert lock_mgr.get_mapping("Vehicle") == {"price": "vehicle_data.amt_inr"}

    # Upstream changes AGAIN: amt_inr is gone, now cost_rupees
    second_drift_payload = {
        "vehicle_data": {
            "cost_rupees": "480000",
        }
    }

    with patch("payload_heal.model.heal_with_llm") as mock_ai:
        mock_ai.return_value = {"price": "vehicle_data.cost_rupees"}

        v3 = Vehicle.model_validate(second_drift_payload)
        mock_ai.assert_called_once()

    assert v3.price == 480000.0

    # Lockfile must now have the NEW path
    assert lock_mgr.get_mapping("Vehicle") == {"price": "vehicle_data.cost_rupees"}


# ==============================================================================
# Additional Tests: @heal_json Decorator
# ==============================================================================

def test_heal_json_standalone_decorator(temp_lockfile: Path):
    """Verifies that @heal_json can wrap ordinary functions and third-party models."""
    from pydantic import BaseModel

    class OrderSchema(BaseModel):
        order_id: int
        total: float

    cfg = HealConfig(lockfile_path=temp_lockfile)

    @heal_json(OrderSchema, config=cfg)
    def process_order(payload: dict):
        order = OrderSchema.model_validate(payload)
        return f"Order #{order.order_id} total: ${order.total:.2f}"

    raw_drifted = {
        "data": {
            "orderId": 5544,
            "total": 129.50,
        }
    }

    result = process_order(raw_drifted)
    assert result == "Order #5544 total: $129.50"
