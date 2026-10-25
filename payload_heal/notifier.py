"""Drift alert notification dispatcher for payload-heal.

Broadcasts schema drift events to configured Slack, Discord, or generic webhooks
so engineering teams can proactively update their upstream scraper or Pydantic models.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from payload_heal.config import HealConfig

logger = logging.getLogger("payload_heal.notifier")


def send_drift_notification(
    model_name: str,
    healed_fields: dict[str, tuple[str, str]],
    config: HealConfig,
) -> bool:
    """Sends a schema drift alert to the configured webhook URL.

    Parameters:
        model_name: The name of the Pydantic model that experienced schema drift.
        healed_fields: Mapping of `field_name -> (healed_source_path, healed_by)`.
                       e.g. `{"price": ("data.pricing.priceUsd", "heuristic")}`.
        config: The active `HealConfig` instance containing `webhook_url`.

    Returns:
        True if the notification was dispatched successfully, False otherwise.
    """
    if not config.webhook_url or not healed_fields:
        return False

    # Format human-readable field summary
    lines: list[str] = [
        f"• `{field}` → `{source_path}` (healed by *{healed_by}*)"
        for field, (source_path, healed_by) in healed_fields.items()
    ]
    fields_bullet_list = "\n".join(lines)

    text_content = (
        f"⚠️ **Payload Schema Drift Healed**\n"
        f"**Model:** `{model_name}`\n"
        f"**Healed Fields ({len(healed_fields)}):**\n"
        f"{fields_bullet_list}\n"
        f"_Lockfile updated automatically. No immediate action required._"
    )

    # Universal webhook payload (compatible with Slack, Discord, and generic webhooks)
    payload: dict[str, Any] = {
        "text": text_content,       # Slack compatibility
        "content": text_content,    # Discord compatibility
        "model": model_name,
        "drift_count": len(healed_fields),
        "healed_fields": {
            field: {"source_path": source_path, "healed_by": healed_by}
            for field, (source_path, healed_by) in healed_fields.items()
        },
    }

    timeout = min(config.timeout_seconds, 3.0)

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(config.webhook_url, json=payload)
            if resp.status_code >= 400:
                logger.warning(
                    "Failed to send drift notification to webhook (%s): HTTP %s - %s",
                    config.webhook_url,
                    resp.status_code,
                    resp.text,
                )
                return False
            logger.info("Successfully sent drift alert for model '%s' to webhook", model_name)
            return True
    except Exception as exc:
        logger.warning(
            "Failed to send drift notification to webhook (%s): %s",
            config.webhook_url,
            exc,
        )
        return False
