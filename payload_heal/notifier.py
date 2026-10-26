"""Drift alert notification dispatcher for payload-heal.

Broadcasts schema drift events to configured Slack, Discord, or generic webhooks
so engineering teams can proactively update their upstream scraper or Pydantic models.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any
from urllib.parse import urlsplit

import httpx

from payload_heal.config import HealConfig

logger = logging.getLogger("payload_heal.notifier")


def sanitize_url_for_logging(url: str) -> str:
    """Masks sensitive tokens, credentials, and query params from a URL before logging."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
        netloc = parts.netloc
        if "@" in netloc:
            _, host_port = netloc.split("@", 1)
            netloc = f"***:***@{host_port}"

        path_segments = [s for s in parts.path.split("/") if s]
        if len(path_segments) > 1:
            masked_path = "/" + "/".join(path_segments[:-1]) + "/***"
        else:
            masked_path = parts.path

        query = "***" if parts.query else ""
        return parts._replace(netloc=netloc, path=masked_path, query=query).geturl()
    except Exception:
        return "[REDACTED_URL]"


def is_safe_webhook_url(
    url: str,
    allowed_hosts: list[str] | None = None,
    allow_internal: bool = False,
) -> tuple[bool, str]:
    """Validates that a webhook URL uses a safe scheme, allowed host, and does not target internal IP addresses.

    Returns:
        `(is_safe, error_reason)`
    """
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL"

    try:
        parsed = urlsplit(url)
    except Exception:
        return False, "Invalid URL structure"

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"Unsupported webhook URL scheme '{scheme}' (must be http or https)"

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in webhook URL"

    # Host allowlist check
    if allowed_hosts:
        hostname_lower = hostname.lower()
        matched = any(
            hostname_lower == h.lower() or hostname_lower.endswith(f".{h.lower()}")
            for h in allowed_hosts
        )
        if not matched:
            return False, f"Host '{hostname}' is not in configured allowed_webhook_hosts"

    if allow_internal:
        return True, ""

    # Check for localhost / loopback / private IP SSRF
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False, f"Destination IP {ip} is an internal or private network address"
        return True, ""
    except ValueError:
        pass  # Hostname is a domain name

    # Reject localhost domain names directly
    hostname_lower = hostname.lower()
    if hostname_lower in ("localhost", "ip6-localhost", "ip6-loopback") or hostname_lower.endswith(".local"):
        return False, f"Destination host '{hostname}' is a local address"

    # DNS resolution check for private IP addresses
    try:
        resolved_ips = socket.getaddrinfo(hostname, None)
        for entry in resolved_ips:
            sockaddr = entry[4]
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False, f"Resolved host IP {ip} is an internal or private network address"
    except Exception as exc:
        logger.warning("DNS resolution check failed for host '%s': %s", hostname, exc)
        return False, f"Unable to resolve host '{hostname}'"

    return True, ""


def send_drift_notification(
    model_name: str,
    healed_fields: dict[str, tuple[str, str]],
    config: HealConfig,
) -> bool:
    """Sends a schema drift alert to the configured webhook URL.

    Parameters:
        model_name: The name or identifier of the Pydantic model that experienced schema drift.
        healed_fields: Mapping of `field_name -> (healed_source_path, healed_by)`.
                       e.g. `{"price": ("data.pricing.priceUsd", "heuristic")}`.
        config: The active `HealConfig` instance containing `webhook_url`.

    Returns:
        True if the notification was dispatched successfully, False otherwise.
    """
    if not config.webhook_url or not healed_fields:
        return False

    is_safe, error_reason = is_safe_webhook_url(
        config.webhook_url,
        allowed_hosts=config.allowed_webhook_hosts,
        allow_internal=config.allow_internal_webhooks,
    )
    if not is_safe:
        logger.warning(
            "Rejected drift webhook dispatch to %s: %s",
            sanitize_url_for_logging(config.webhook_url),
            error_reason,
        )
        return False

    safe_url = sanitize_url_for_logging(config.webhook_url)

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
                    "Failed to send drift notification to webhook (%s): HTTP %s",
                    safe_url,
                    resp.status_code,
                )
                return False
            logger.info("Successfully sent drift alert for model '%s' to webhook", model_name)
            return True
    except Exception as exc:
        logger.warning(
            "Failed to send drift notification to webhook (%s): %s",
            safe_url,
            exc,
        )
        return False
