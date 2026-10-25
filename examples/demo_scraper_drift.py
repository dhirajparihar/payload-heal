"""Self-contained demo simulating an e-commerce scraper surviving 3 schema drifts.

Demonstrates:
1. v1 Original: Fast Path (0ms, $0 cost)
2. v2 Minor Casing/Nesting Drift: Tier-1 Heuristic Matcher (<1ms, $0 cost)
3. v3 Complete Backend Rewrite: Tier-2 AI Skeleton Healer (runs once, caches forever)
4. v3 Repeat: Tier-0 Lockfile Cache (0ms, $0 cost)
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import ClassVar
from unittest.mock import patch

from pydantic import Field

from payload_heal import HealConfig, HealableModel
from payload_heal.lockfile import LockFileManager

DEMO_LOCKFILE = Path(".demo_heal_lock.json")


class Product(HealableModel):
    """Pydantic model representing our e-commerce product schema."""

    heal_config: ClassVar[HealConfig] = HealConfig(
        lockfile_path=DEMO_LOCKFILE,
        enable_heuristics=True,
        enable_ai=True,
    )

    title: str = Field(description="Display title of the product")
    price: float = Field(description="Current retail price in USD")
    rating: float = Field(description="User review rating out of 5.0")
    in_stock: bool = Field(description="Whether the product is currently in stock")


def print_header(title: str) -> None:
    print("\n" + "=" * 85)
    print(f"  {title}")
    print("=" * 85)


def print_table(results: list[dict[str, str]]) -> None:
    print(f"\n{'Iteration':<12} {'API State':<30} {'Tier Used':<22} {'Latency':<10} {'Status'}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['iteration']:<12} {r['state']:<30} {r['tier']:<22} {r['time']:<10} {r['status']}"
        )
    print("-" * 85)


def run_demo() -> None:
    # Ensure fresh lockfile for the demo
    if DEMO_LOCKFILE.exists():
        DEMO_LOCKFILE.unlink()

    results: list[dict[str, str]] = []

    print_header("PAYLOAD-HEAL LIVE DEMO: Resilient Web Scraper Ingestion")
    print("Target Schema:")
    print("  title: str, price: float, rating: float, in_stock: bool")

    # --------------------------------------------------------------------------
    # Step 1: v1 Original API Response
    # --------------------------------------------------------------------------
    print("\n[Stage 1] Ingesting v1 (Original Payload)...")
    payload_v1 = {
        "title": "Sony WH-1000XM5 Wireless Headphones",
        "price": 399.99,
        "rating": 4.8,
        "in_stock": True,
    }

    t0 = time.perf_counter()
    p1 = Product.model_validate(payload_v1)
    dt1 = (time.perf_counter() - t0) * 1000

    results.append({
        "iteration": "Run 1",
        "state": "v1 (Normal Schema)",
        "tier": "Fast Path (0ms, $0)",
        "time": f"{dt1:.3f} ms",
        "status": "[OK] Ingested",
    })
    print(f"  Result: {p1.title} | ${p1.price} | {p1.rating}/5.0 | in_stock={p1.in_stock}")

    # --------------------------------------------------------------------------
    # Step 2: v2 Minor Casing / Nesting Drift (Heuristic Matcher)
    # --------------------------------------------------------------------------
    print("\n[Stage 2] Upstream Drifted to v2 (camelCase + Nested Wrappers)...")
    payload_v2 = {
        "status": "success",
        "data": {
            "itemDetails": {
                "itemTitle": "Sony WH-1000XM5 Wireless Headphones",
                "priceUsd": 399.99,
                "userRating": 4.8,
                "inStock": True,
            }
        },
    }

    t0 = time.perf_counter()
    p2 = Product.model_validate(payload_v2)
    dt2 = (time.perf_counter() - t0) * 1000

    results.append({
        "iteration": "Run 2",
        "state": "v2 (camelCase + Nesting)",
        "tier": "Tier-1 Heuristic ($0)",
        "time": f"{dt2:.3f} ms",
        "status": "[OK] Healed",
    })
    print(f"  Result: {p2.title} | ${p2.price} | {p2.rating}/5.0 | in_stock={p2.in_stock}")

    # --------------------------------------------------------------------------
    # Step 3: v3 Complete Backend Rewrite (AI Skeleton Healer)
    # --------------------------------------------------------------------------
    print("\n[Stage 3] Upstream Drifted to v3 (Radical Redesign / Microservice)...")
    payload_v3 = {
        "catalog_response": {
            "listing_node": {
                "headline_label": "Sony WH-1000XM5 Wireless Headphones",
                "retail_amount": "399.99",
                "average_score": "4.8",
                "inventory_available": "true",
            }
        }
    }

    # In a live environment with an API key, this calls OpenAI/Groq/Ollama.
    # For this instant runnable demo, we simulate the LLM response mapping:
    ai_mock_mapping = {
        "title": "catalog_response.listing_node.headline_label",
        "price": "catalog_response.listing_node.retail_amount",
        "rating": "catalog_response.listing_node.average_score",
        "in_stock": "catalog_response.listing_node.inventory_available",
    }

    with patch("payload_heal.model.heal_with_llm", return_value=ai_mock_mapping):
        t0 = time.perf_counter()
        p3 = Product.model_validate(payload_v3)
        dt3 = (time.perf_counter() - t0) * 1000

    results.append({
        "iteration": "Run 3",
        "state": "v3 (Complete Rewrite)",
        "tier": "Tier-2 AI Healer",
        "time": f"{dt3:.3f} ms",
        "status": "[OK] Healed & Cached",
    })
    print(f"  Result: {p3.title} | ${p3.price} | {p3.rating}/5.0 | in_stock={p3.in_stock}")

    # --------------------------------------------------------------------------
    # Step 4: v3 Repeat Call (Tier 0 Lockfile Cache)
    # --------------------------------------------------------------------------
    print("\n[Stage 4] Ingesting v3 Again (Testing Tier 0 Lockfile Cache hit)...")
    with patch("payload_heal.model.heal_with_llm") as mock_ai:
        t0 = time.perf_counter()
        p4 = Product.model_validate(payload_v3)
        dt4 = (time.perf_counter() - t0) * 1000
        # Assert LLM was never called because lockfile cached the mapping
        mock_ai.assert_not_called()

    results.append({
        "iteration": "Run 4",
        "state": "v3 Repeat Payload",
        "tier": "Tier-0 Lockfile (0ms)",
        "time": f"{dt4:.3f} ms",
        "status": "[OK] 0ms Cache Hit",
    })
    print(f"  Result: {p4.title} | ${p4.price} | {p4.rating}/5.0 | in_stock={p4.in_stock}")

    # --------------------------------------------------------------------------
    # Final Summary Table
    # --------------------------------------------------------------------------
    print_header("PERFORMANCE & HEALING SUMMARY")
    print_table(results)

    # Show lockfile contents
    print("\n[Persisted Lockfile Cache]: .demo_heal_lock.json")
    if DEMO_LOCKFILE.exists():
        lock_mgr = LockFileManager(DEMO_LOCKFILE)
        raw = lock_mgr._read_raw()
        print(json.dumps(raw, indent=2))
        DEMO_LOCKFILE.unlink()  # cleanup demo lockfile
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_demo()
