"""Skeleton extraction and safe dot/bracket notation path traversal for JSON payloads.

This module provides tools to:
1. Extract a privacy-safe, lightweight schema skeleton from any nested JSON/GraphQL
   payload, automatically masking PII (emails, bearer tokens, credit cards).
2. Traverse nested payloads using dot and bracket notation (e.g. data.items[0].pricing.amount)
   with precise detection of present falsy values (0, False, empty string, None).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

# ==============================================================================
# PII Detection and Masking Patterns
# ==============================================================================

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

_BEARER_RE = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9\-_.~+/]+=*",
)

_BASIC_AUTH_RE = re.compile(
    r"(?i)\bbasic\s+[A-Za-z0-9+/=]{8,}",
)

_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.?[A-Za-z0-9-_]*"
)

_API_KEY_RE = re.compile(
    r"\b(?:sk|ghp|gho|glpat|xoxb|xoxp|key|token|secret)[-_][A-Za-z0-9_-]{12,}\b"
)

_AWS_KEY_RE = re.compile(
    r"\bAKIA[0-9A-Z]{16}\b"
)

_SSN_RE = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"
)

_SENSITIVE_KEY_PATTERNS = re.compile(
    r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|private[_-]?key|bearer|credentials?|ssn|social[_-]?security|cvv|cvc|pin|card[_-]?number|credit[_-]?card|token|session|cookie)"
)

# Standard credit card formats: 16-digit cards with separators or 15-digit Amex
_CC_FORMATTED_RE = re.compile(
    r"\b(?:\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}|\d{4}[ -]\d{6}[ -]\d{5})\b"
)

# Contiguous 13-19 digit candidate numbers
_CC_CANDIDATE_RE = re.compile(
    r"\b\d{13,19}\b"
)


def is_luhn_valid(card_num: str) -> bool:
    """Verifies whether a numeric sequence passes the Luhn checksum algorithm."""
    digits = [int(c) for c in card_num if c.isdigit()]
    if len(digits) < 2:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def mask_pii(text: str) -> str:
    """Masks emails, bearer/API tokens, SSNs, private keys, and credit card numbers from strings.

    Note: Masking is best-effort redaction and does not provide an absolute guarantee of anonymity.
    """
    if not isinstance(text, str):
        text = str(text)

    # Mask private key blocks
    text = _PRIVATE_KEY_RE.sub("[MASKED_PRIVATE_KEY]", text)
    # Mask bearer tokens and headers
    text = _BEARER_RE.sub("Bearer [MASKED_TOKEN]", text)
    # Mask basic auth credentials
    text = _BASIC_AUTH_RE.sub("Basic [MASKED_TOKEN]", text)
    # Mask JWTs
    text = _JWT_RE.sub("[MASKED_TOKEN]", text)
    # Mask API keys
    text = _API_KEY_RE.sub("[MASKED_TOKEN]", text)
    # Mask AWS Access Keys
    text = _AWS_KEY_RE.sub("[MASKED_AWS_KEY]", text)
    # Mask SSNs
    text = _SSN_RE.sub("[MASKED_SSN]", text)
    # Mask email addresses
    text = _EMAIL_RE.sub("[MASKED_EMAIL]", text)
    # Mask formatted credit cards
    text = _CC_FORMATTED_RE.sub("[MASKED_CREDIT_CARD]", text)

    # Mask contiguous credit card numbers (13-19 digits) validated by Luhn algorithm
    def _mask_luhn(match: re.Match[str]) -> str:
        val = match.group(0)
        digits = [c for c in val if c.isdigit()]
        if 13 <= len(digits) <= 19 and is_luhn_valid(val):
            return "[MASKED_CREDIT_CARD]"
        return val

    text = _CC_CANDIDATE_RE.sub(_mask_luhn, text)
    return text


def sanitize_data_recursively(data: Any, max_depth: int = 5) -> Any:
    """Recursively sanitizes nested dictionaries, lists, and primitives.

    - Replaces sensitive dictionary keys' values with '[MASKED_SECRET]'
    - Masks PII patterns (emails, tokens, API keys, SSNs, credit cards) in all strings
    - Normalizes Unicode homoglyphs and compatibility forms in key names
    - Recursively traverses nested structures up to max_depth
    """
    if max_depth <= 0:
        return "[TRUNCATED]"

    if isinstance(data, dict):
        sanitized: dict[str, Any] = {}
        for k, v in data.items():
            k_str = str(k)
            # Normalize Unicode to detect homoglyphs and diacritics
            norm_k = unicodedata.normalize("NFKD", k_str)
            if _SENSITIVE_KEY_PATTERNS.search(k_str) or _SENSITIVE_KEY_PATTERNS.search(norm_k):
                sanitized[k_str] = "[MASKED_SECRET]"
            else:
                sanitized[k_str] = sanitize_data_recursively(v, max_depth - 1)
        return sanitized

    if isinstance(data, list):
        return [sanitize_data_recursively(item, max_depth - 1) for item in data]

    if isinstance(data, str):
        return mask_pii(data)

    if isinstance(data, (int, float, bool)) or data is None:
        return data

    return mask_pii(str(data))


def format_sample_preview(val: Any, max_len: int = 80) -> str:
    """Formats a value into a safe, PII-masked, recursively sanitized preview string."""
    sanitized_val = sanitize_data_recursively(val)

    if sanitized_val is None:
        return "null"
    if isinstance(sanitized_val, bool):
        return "true" if sanitized_val else "false"
    if isinstance(sanitized_val, (int, float)):
        return str(sanitized_val)
    if isinstance(sanitized_val, str):
        if len(sanitized_val) > max_len:
            return f"{sanitized_val[:max_len]}..."
        return sanitized_val
    if isinstance(sanitized_val, dict):
        if not sanitized_val:
            return "{}"
        try:
            preview_dict = {
                str(k): v for k, v in list(sanitized_val.items())[:2]
            }
            s = json.dumps(preview_dict)
            if len(s) > max_len:
                return f"{s[:max_len]}..."
            return s
        except Exception:
            return "{...}"
    if isinstance(sanitized_val, list):
        if not sanitized_val:
            return "[]"
        sample_items: list[str] = []
        for x in sanitized_val[:2]:
            if isinstance(x, str):
                sample_items.append(repr(x))
            elif isinstance(x, bool):
                sample_items.append("true" if x else "false")
            elif isinstance(x, (int, float)):
                sample_items.append(str(x))
            elif x is None:
                sample_items.append("null")
            else:
                sample_items.append(format_sample_preview(x, max_len=20))
        if len(sanitized_val) > len(sample_items):
            s = f"[{', '.join(sample_items)}, ...]"
        else:
            s = f"[{', '.join(sample_items)}]"
        if len(s) > max_len:
            return f"{s[:max_len]}..."
        return s
    return str(sanitized_val)[:max_len]


# ==============================================================================
# Skeleton Extraction
# ==============================================================================

def extract_json_skeleton(
    data: Any,
    prefix: str = "",
    max_list_sample: int = 1,
) -> dict[str, dict[str, Any]]:
    """Recursively flattens any nested dict/list JSON payload into a mapping of:
    dot_path -> {"type": type_name, "sample": truncated_preview_str}

    Handles:
    - Nested dicts (`a.b.c`)
    - Lists of primitives (`tags`)
    - Lists of dicts (`items[0].price`)
    - Empty collections (`{}`, `[]`)
    - Falsy values (`0`, `False`, `""`, `None`)
    - Automatic PII masking for emails, tokens, and credit cards in preview samples.

    Parameters:
        data: The JSON-like dictionary, list, or primitive to analyze.
        prefix: Current path prefix (used in recursive calls or root namespaces).
        max_list_sample: Maximum number of list elements to inspect for structure.

    Returns:
        Mapping of dot/bracket paths to type name and truncated sample preview.
    """
    skeleton: dict[str, dict[str, Any]] = {}

    def _infer_primitive_type(val: Any) -> str:
        if val is None:
            return "null"
        if isinstance(val, bool):
            return "bool"
        if isinstance(val, int):
            return "int"
        if isinstance(val, float):
            return "float"
        if isinstance(val, str):
            return "str"
        return type(val).__name__

    def _infer_list_type(lst: list[Any]) -> str:
        if not lst:
            return "list"
        elem_types = {_infer_primitive_type(x) for x in lst[:max_list_sample]}
        if len(elem_types) == 1:
            return f"list[{elem_types.pop()}]"
        return "list"

    def _recurse(curr: Any, curr_path: str) -> None:
        if isinstance(curr, dict):
            if curr_path:
                skeleton[curr_path] = {
                    "type": "dict",
                    "sample": format_sample_preview(curr, max_len=60) if curr else "{}",
                }
            if not curr:
                return
            for key, val in curr.items():
                key_str = str(key)
                if key_str.isidentifier():
                    next_path = f"{curr_path}.{key_str}" if curr_path else key_str
                else:
                    escaped_key = key_str.replace("\\", "\\\\").replace("'", "\\'")
                    next_path = f"{curr_path}['{escaped_key}']" if curr_path else f"['{escaped_key}']"
                _recurse(val, next_path)

        elif isinstance(curr, list):
            if not curr:
                if curr_path:
                    skeleton[curr_path] = {
                        "type": "list",
                        "sample": "[]",
                    }
                return

            # Check if list contains dicts or nested lists
            has_nested = any(isinstance(x, (dict, list)) for x in curr[:max_list_sample])
            if has_nested:
                for idx, item in enumerate(curr[:max_list_sample]):
                    next_path = f"{curr_path}[{idx}]" if curr_path else f"[{idx}]"
                    _recurse(item, next_path)
            else:
                # List of primitives
                list_type = _infer_list_type(curr)
                sample_str = format_sample_preview(curr[:max_list_sample])
                if len(curr) > max_list_sample and not sample_str.endswith("...]"):
                    if sample_str.endswith("]"):
                        sample_str = sample_str[:-1] + ", ...]"
                skeleton[curr_path] = {
                    "type": list_type,
                    "sample": sample_str,
                }
        else:
            # Primitive leaf value
            skeleton[curr_path] = {
                "type": _infer_primitive_type(curr),
                "sample": format_sample_preview(curr),
            }

    _recurse(data, prefix)
    return skeleton


# ==============================================================================
# Dot and Bracket Path Traversal
# ==============================================================================

def parse_dot_path(path: str) -> list[str | int] | None:
    """Parses a dot and bracket notation path into a sequence of keys and indices.

    Supports:
    - Dot notation: `a.b.c`
    - Array indices: `items[0]`, `matrix[1][2]`
    - Root array indices: `[0].title`
    - Quoted keys inside brackets: `users['first.name']`, `data["key with space"]`

    Returns:
        List of str/int tokens, or None if the path syntax is invalid.
    """
    if not isinstance(path, str) or not path.strip():
        return None

    tokens: list[str | int] = []
    i = 0
    n = len(path)

    while i < n:
        char = path[i]

        if char == ".":
            # Dots must connect valid tokens
            if i == 0 or i == n - 1 or path[i - 1] == "." or path[i + 1] in (".", "["):
                return None
            i += 1
            continue

        if char == "[":
            i += 1
            # Skip optional whitespace
            while i < n and path[i].isspace():
                i += 1
            if i >= n:
                return None

            quote_char = path[i]
            if quote_char in ("'", '"'):
                i += 1
                token_chars: list[str] = []
                closed_quote = False
                while i < n:
                    c = path[i]
                    if c == "\\":
                        if i + 1 < n:
                            next_c = path[i + 1]
                            if next_c in ("'", '"', "\\"):
                                token_chars.append(next_c)
                                i += 2
                                continue
                            else:
                                token_chars.append(c)
                                i += 1
                                continue
                        else:
                            return None
                    elif c == quote_char:
                        closed_quote = True
                        i += 1
                        break
                    else:
                        token_chars.append(c)
                        i += 1

                if not closed_quote:
                    return None

                while i < n and path[i].isspace():
                    i += 1
                if i >= n or path[i] != "]":
                    return None
                i += 1
                tokens.append("".join(token_chars))
            else:
                bracket_start = i
                while i < n and path[i] != "]":
                    i += 1
                if i >= n:
                    return None
                content = path[bracket_start:i].strip()
                i += 1
                if not content:
                    return None
                if content.isdigit():
                    tokens.append(int(content))
                elif content.startswith("-") and content[1:].isdigit():
                    tokens.append(int(content))
                elif content.isidentifier():
                    tokens.append(content)
                else:
                    return None
        else:
            # Identifier segment up to next '.' or '['
            start = i
            while i < n and path[i] not in (".", "["):
                if path[i] == "]":
                    return None
                i += 1
            segment = path[start:i]
            if not segment:
                return None
            tokens.append(segment)

    return tokens


def get_by_dot_path(data: Any, path: str) -> tuple[bool, Any]:
    """Safely traverses nested dicts and lists using dot and bracket notation.

    Example paths:
    - `"data.items[0].pricing.amount"`
    - `"user.profile.emails[0]"`
    - `"matrix[0][1]"`

    Returns:
        (True, value) if the path was successfully reached, even if the value is
        falsy (`False`, `0`, `""`, `None`, `{}` or `[]`).
        (False, None) if the path does not exist in the structure.
    """
    tokens = parse_dot_path(path)
    if tokens is None:
        return False, None

    current = data
    for token in tokens:
        if isinstance(token, int):
            if isinstance(current, (list, tuple)):
                if 0 <= token < len(current):
                    current = current[token]
                elif token < 0 and -len(current) <= token:
                    current = current[token]
                else:
                    return False, None
            elif isinstance(current, dict):
                if token in current:
                    current = current[token]
                elif str(token) in current:
                    current = current[str(token)]
                else:
                    return False, None
            else:
                return False, None
        else:
            if isinstance(current, dict):
                if token in current:
                    current = current[token]
                else:
                    return False, None
            elif isinstance(current, (list, tuple)) and token.isdigit():
                idx = int(token)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return False, None
            elif hasattr(current, token) and not token.startswith("_"):
                current = getattr(current, token)
            else:
                return False, None

    return True, current
