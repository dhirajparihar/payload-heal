<div align="center">

# 🩹 payload-heal

**Self-Healing Pydantic Models for Web Scrapers, GraphQL Ingestion, and Volatile JSON APIs.**

[![PyPI version](https://img.shields.io/pypi/v/payload-heal.svg?color=blue)](https://pypi.org/project/payload-heal/)
[![Python versions](https://img.shields.io/pypi/pyversions/payload-heal.svg)](https://pypi.org/project/payload-heal/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-e92063.svg)](https://docs.pydantic.dev/)
[![CI](https://github.com/dhirajparihar/payload-heal/actions/workflows/ci.yml/badge.svg)](https://github.com/dhirajparihar/payload-heal/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-67%20passed-brightgreen.svg)](https://github.com/dhirajparihar/payload-heal/actions)

*Resilient Pydantic v2 model recovery from renamed keys, casing drift, and nested wrapper mutations.*

</div>

---

## ⚡ The Overview

Web scrapers and upstream API integrations **break unexpectedly** when an upstream provider renames `price_usd` to `productPrice`, wraps fields inside `data.listing.item`, or migrates to a revised GraphQL schema.

Calling an LLM on every incoming payload is impractical: it adds hundreds of milliseconds of latency, incurs compounding API costs, and risks leaking sensitive data to external endpoints.

**`payload-heal`** provides a drop-in replacement for `pydantic.BaseModel` called **`HealableModel`** with a **3-tier hybrid self-healing engine**:
- **Near-zero overhead (<0.1ms)** when incoming payloads are healthy.
- **Fast, deterministic heuristic auto-healing ($0 cost)** for common schema drifts (casing changes, nested leaf promotion, token overlap).
- **One-time AI recovery**: when heuristics cannot resolve an ambiguous drift, queries an LLM once on a lightweight, PII-sanitized skeleton, persists the verified mapping into a local lockfile cache, and **subsequently serves it from cache**.

---

## 🧠 How the 3-Tier Hybrid Engine Works

```text
Incoming JSON / GraphQL Payload
              │
              ▼
    [ All Required Fields Present? ]
         ├─── YES ───► [ Fast Path: <0.1ms Overhead, $0 Cost ] ───► Validated Model
         │
         └─── NO  (Schema Drift Detected!)
              │
              ├──► Tier 0: Lockfile Cache (.payload_heal_lock.json)
              │         Lookup previously healed dot paths (<0.1ms, $0)
              │         *Re-Drift Guard: Auto-invalidates if upstream drifts a second time!
              │         *Type Safety: Verifies cached values against model annotations.
              │
              ├──► Tier 1: Deterministic Heuristic Matcher
              │         Zero-LLM instant recovery (~1ms, $0)
              │         - camelCase / snake_case / kebab-case normalization
              │         - Nested leaf promotion (e.g., user_id -> data.user.id)
              │         - Global 1-to-1 candidate assignment across competing fields
              │         - Ambiguity protection (refuses to guess on tied confidence)
              │
              └──► Tier 2: AI Skeleton Healer
                        Invoked ONLY ONCE when heuristics cannot resolve drift.
                        - Generates a compact, recursively sanitized JSON skeleton
                        - Queries OpenAI / Groq / Ollama / Gemini for path mapping
                        - Strict Type-Safety verification prevents cache poisoning
                        - Caches result in Tier 0 lockfile and alerts team via Webhook
```

---

## 🚀 30-Second Quickstart

### Installation

```bash
pip install payload-heal
```

### Drop-In Usage

Simply replace `pydantic.BaseModel` with `HealableModel`:

```python
from pydantic import Field
from payload_heal import HealableModel

class Product(HealableModel):
    title: str
    price: float = Field(description="Retail price in USD")
    rating: float
    in_stock: bool

# Upstream scraped API drifted: keys renamed and wrapped inside nested objects!
broken_payload = {
    "status": "ok",
    "data": {
        "itemDetails": {
            "itemTitle": "Sony WH-1000XM5 Wireless Headphones",
            "priceUsd": "399.99",   # string coerced to float
            "userRating": 4.8,
            "inStock": "true",       # string coerced to bool
        }
    }
}

# Automatically recovers!
product = Product.model_validate(broken_payload)

print(product.title)     # "Sony WH-1000XM5 Wireless Headphones"
print(product.price)     # 399.99 (float)
print(product.rating)    # 4.8 (float)
print(product.in_stock)  # True (bool)

# Next time this payload is encountered: Tier 0 Lockfile Cache hit in 0ms!
```

---

## 🔌 Supported AI Providers (Tier 2)

Configure providers globally or per-model via `HealConfig`:

```python
from typing import ClassVar
from payload_heal import HealableModel, HealConfig

# 1. OpenAI (Default: gpt-4o-mini)
class OpenAIModel(HealableModel):
    heal_config: ClassVar[HealConfig] = HealConfig(
        llm_provider="openai",
        api_key="sk-..."  # or set OPENAI_API_KEY env var
    )

# 2. Groq (Ultra-fast Llama-3.3-70B)
class GroqModel(HealableModel):
    heal_config: ClassVar[HealConfig] = HealConfig(
        llm_provider="groq",
        model="llama-3.3-70b-versatile",
        api_key="gsk_..."  # or set GROQ_API_KEY env var
    )

# 3. Local Ollama (100% Free & Private on localhost)
class OllamaModel(HealableModel):
    heal_config: ClassVar[HealConfig] = HealConfig(
        llm_provider="ollama",
        model="llama3",
        api_base="http://localhost:11434/v1"
    )

# 4. Google Gemini
class GeminiModel(HealableModel):
    heal_config: ClassVar[HealConfig] = HealConfig(
        llm_provider="gemini",
        api_key="..."  # or set GEMINI_API_KEY env var
    )
```

---

## 🛠️ Standalone Decorator (`@heal_json`)

Wrap existing third-party models or function handlers without modifying class definitions:

```python
from pydantic import BaseModel
from payload_heal import heal_json

class ThirdPartySchema(BaseModel):
    user_id: int
    email: str

@heal_json(ThirdPartySchema)
def ingest_webhook(payload: dict):
    # payload is healed before function execution
    user = ThirdPartySchema.model_validate(payload)
    print(f"Processed user #{user.user_id}: {user.email}")
```

---

## 🛡️ Privacy & Security Safeguards

### Recursive Secret & PII Sanitization
Before any payload preview is processed or sent to an external LLM endpoint, `payload-heal` executes recursive sanitization:
- **Sensitive Key Masking**: Dictionary keys indicating secrets (`password`, `api_key`, `token`, `secret`, `private_key`, `credentials`, `ssn`, `cvv`) automatically have their values redacted to `"[MASKED_SECRET]"`.
- **Credential & Token Regexes**: API keys (`sk-`, `ghp-`, `glpat-`, `xoxb-`), AWS Access Keys (`AKIA...`), JWT tokens, and Bearer authorization headers are masked.
- **Personal Identifiers**: Email addresses, US SSNs, and formatted or Luhn-valid credit card numbers are masked.
- **Recursive Traversal**: Sanitization recurses through nested dictionaries and lists, ensuring deeply nested secrets in previews do not escape.

> **Note on Privacy Guarantees**: Masking is best-effort pattern and heuristic redaction. For sensitive environments requiring strict compliance, use local providers like **Ollama** (`llm_provider="ollama"`) or disable AI calls entirely (`enable_ai=False`).

### Webhook SSRF Protection
Outbound schema drift notifications enforce security controls:
- **Protocol Restrictions**: Only `http` and `https` schemes are accepted.
- **Private & Internal IP Blocking**: Destination IPs resolving to loopback (`127.0.0.1`), private networks (`10.0.0.0/8`, `192.168.0.0/16`, `172.16.0.0/12`), link-local (`169.254.169.254`), or `localhost` are blocked by default to prevent Server-Side Request Forgery (SSRF).
- **Host Allowlisting**: Use `HealConfig(allowed_webhook_hosts=["hooks.slack.com", "discord.com"])` to restrict webhook destinations to authorized hosts.
- **Sanitized Logging**: Webhook path tokens, query parameters, and credentials are redacted in logs. Raw response bodies are omitted.

### AI Prompt Injection Hardening
Payload paths and preview samples are treated as untrusted data, enclosed within explicit `<untrusted_payload_skeleton>` boundaries, and accompanied by strict system instructions to reject prompt overrides embedded within candidate samples.

---

## ⚙️ Operational & Deployment Guide

### Read-Only Filesystems (AWS Lambda / Serverless)
By default, `payload-heal` writes cached mappings to `.payload_heal_lock.json`. In serverless environments with a read-only root filesystem, point the lockfile to a writable directory such as `/tmp`:
```python
HealConfig(lockfile_path=Path("/tmp/.payload_heal_lock.json"))
```

### Multi-Process Concurrency
The `LockFileManager` ensures thread safety via `threading.RLock` and inter-process concurrency through atomic file locks (utilizing the cross-platform `filelock` package or native `msvcrt`/`fcntl` OS primitives). Updates are written to temporary files and atomically replaced via `os.replace`.

### Offline & Air-Gapped Deployments
To run completely offline with zero external network requests:
```python
# Pure deterministic heuristics only ($0 cost, 0 external calls)
HealConfig(enable_ai=False)
```

### Model Cache Namespacing
Lockfile entries are namespaced by module-qualified model identifiers (e.g. `myapp.models.Product`), preventing cached mapping collisions between models that share the same class name across different modules.

---

## ⚠️ Limitations & Fail-Safe Defaults

- **Fail-Safe by Default**: If a candidate mapping is ambiguous, fails type compatibility against the Pydantic field annotation, or cannot be resolved, `payload-heal` rejects the mapping and allows standard Pydantic validation errors to be raised visibly.
- **Ambiguity Refusal**: If multiple candidate paths tie in heuristic confidence within the configured margin (default: 0.05), the matcher refuses to guess to avoid data corruption.
- **Best-Effort Heuristics**: Heuristic matching relies on structural casing, token Jaccard overlap, and sequence similarity. Radically transformed semantic drifts require Tier-2 AI or manual model updates.

---

## 🔔 Slack / Discord Webhook Drift Alerts

Get notified in Slack or Discord the moment an upstream API drifts:

```python
class MonitoredModel(HealableModel):
    heal_config: ClassVar[HealConfig] = HealConfig(
        webhook_url="https://hooks.slack.com/services/T00/B00/X00",
        allowed_webhook_hosts=["hooks.slack.com"]
    )
    price: float
```

When schema drift is resolved, `payload-heal` fires a non-blocking notification:
```text
⚠️ Payload Schema Drift Healed
Model: MonitoredModel
Healed Fields (1):
• price → data.pricing.priceUsd (healed by heuristic)
Lockfile updated automatically. No immediate action required.
```

---

## 💻 Developer CLI

Inspect skeletons and manage lockfiles from your terminal:

```bash
# View all healed models and field mappings
payload-heal status

# Inspect a raw JSON file and view skeleton structure
payload-heal inspect scraper_dump.json

# Reset cached mappings for a specific model (or all models)
payload-heal reset ProductModel
payload-heal reset
```

---

## 🧪 Test Suite

Run unit and integration tests across all modules:

```bash
pytest -v
```
All **67 tests** pass with coverage across heuristics, recursive PII redaction, AI healing, lockfile concurrency, SSRF protection, and end-to-end drift scenarios.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details. Built for resilient engineering teams.
