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


def is_safe_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Returns True if an IP is globally routable and not internal/loopback/private."""
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


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
        if not is_safe_ip(ip):
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
            if not is_safe_ip(ip):
                return False, f"Resolved host IP {ip} is an internal or private network address"
    except Exception as exc:
        logger.warning("DNS resolution check failed for host '%s': %s", hostname, exc)
        return False, f"Unable to resolve host '{hostname}'"

    return True, ""


# ==============================================================================
# SSRF-Safe Request-Time Socket Backend & Transport
# ==============================================================================

import httpcore
from httpcore._backends.sync import SyncBackend, SyncStream
from httpcore._exceptions import ConnectError, ConnectTimeout, ExceptionMapping, map_exceptions
from urllib.parse import urljoin


class SSRFSafeBackend(SyncBackend):
    """Custom HTTPCore network backend that enforces IP safety at connection time.

    Defends against DNS rebinding (TOCTOU) attacks by resolving and verifying the
    exact IP address right when opening the TCP socket, while allowing TLS to verify
    the original hostname.
    """

    def __init__(
        self,
        allowed_hosts: list[str] | None = None,
        allow_internal: bool = False,
    ) -> None:
        self.allowed_hosts = allowed_hosts
        self.allow_internal = allow_internal

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> SyncStream:
        if socket_options is None:
            socket_options = []

        # Enforce host allowlist at socket connection time
        if self.allowed_hosts:
            host_lower = host.lower()
            matched = any(
                host_lower == h.lower() or host_lower.endswith(f".{h.lower()}")
                for h in self.allowed_hosts
            )
            if not matched:
                raise ConnectError(f"Host '{host}' is not in allowed webhook hosts list")

        if not self.allow_internal:
            # Resolve destination IP at the moment of connection
            try:
                addr_info = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
            except Exception as exc:
                raise ConnectError(f"DNS resolution failed for '{host}': {exc}") from exc

            safe_addr: tuple[str, int] | None = None
            for family, socktype, proto, canonname, sockaddr in addr_info:
                ip_str = sockaddr[0]
                try:
                    ip = ipaddress.ip_address(ip_str)
                    if not is_safe_ip(ip):
                        continue
                    safe_addr = (ip_str, sockaddr[1])
                    break
                except ValueError:
                    continue

            if safe_addr is None:
                raise ConnectError(
                    f"Connection blocked: host '{host}' resolved to private/internal IP address(es) at connection time"
                )

            target_address = safe_addr
        else:
            target_address = (host, port)

        source_address = None if local_address is None else (local_address, 0)
        exc_map: ExceptionMapping = {
            socket.timeout: ConnectTimeout,
            OSError: ConnectError,
        }

        with map_exceptions(exc_map):
            sock = socket.create_connection(
                target_address,
                timeout,
                source_address=source_address,
            )
            for option in socket_options:
                sock.setsockopt(*option)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return SyncStream(sock)


class SSRFSafeTransport(httpx.HTTPTransport):
    """HTTPX Transport powered by SSRFSafeBackend to protect against DNS rebinding."""

    def __init__(
        self,
        allowed_hosts: list[str] | None = None,
        allow_internal: bool = False,
        **kwargs: Any,
    ) -> None:
        self.allowed_hosts = allowed_hosts
        self.allow_internal = allow_internal
        super().__init__(**kwargs)

        # Resolve SSL context robustly across HTTPX/HTTPCore versions
        ssl_ctx = getattr(self._pool, "_ssl_context", None)
        if ssl_ctx is None:
            verify = kwargs.get("verify", True)
            cert = kwargs.get("cert")
            trust_env = kwargs.get("trust_env", True)
            try:
                ssl_ctx = httpx.create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)
            except Exception:
                ssl_ctx = None

        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl_ctx,
            network_backend=SSRFSafeBackend(
                allowed_hosts=allowed_hosts,
                allow_internal=allow_internal,
            ),
        )


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
        transport = SSRFSafeTransport(
            allowed_hosts=config.allowed_webhook_hosts,
            allow_internal=config.allow_internal_webhooks,
        )
        with httpx.Client(transport=transport, timeout=timeout, follow_redirects=False) as client:
            current_url = config.webhook_url
            for redirect_step in range(4):  # Up to 3 redirects allowed
                resp = client.post(current_url, json=payload)
                if resp.is_redirect:
                    location = resp.headers.get("Location")
                    if not location:
                        logger.warning("Redirect received from %s with no Location header", safe_url)
                        return False
                    next_url = urljoin(current_url, location)
                    redir_safe, redir_err = is_safe_webhook_url(
                        next_url,
                        allowed_hosts=config.allowed_webhook_hosts,
                        allow_internal=config.allow_internal_webhooks,
                    )
                    if not redir_safe:
                        logger.warning(
                            "Blocked unsafe redirect from %s to %s: %s",
                            safe_url,
                            sanitize_url_for_logging(next_url),
                            redir_err,
                        )
                        return False
                    current_url = next_url
                    continue

                if resp.status_code >= 400:
                    logger.warning(
                        "Failed to send drift notification to webhook (%s): HTTP %s",
                        safe_url,
                        resp.status_code,
                    )
                    return False

                logger.info("Successfully sent drift alert for model '%s' to webhook", model_name)
                return True

            logger.warning("Too many redirects attempted when posting to webhook %s", safe_url)
            return False
    except Exception as exc:
        logger.warning(
            "Failed to send drift notification to webhook (%s): %s",
            safe_url,
            exc,
        )
        return False
