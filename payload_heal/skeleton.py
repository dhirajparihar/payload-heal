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

_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.?[A-Za-z0-9-_]*"
)

_API_KEY_RE = re.compile(
    r"\b(?:sk|ghp|gho|glpat|xoxb|xoxp|key)-[A-Za-z0-9_-]{16,}\b"
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
    """Masks emails, bearer/API tokens, and credit card numbers from preview strings.

    Returns sanitized text suitable for telemetry or AI skeleton prompts.
    """
    if not isinstance(text, str):
        text = str(text)

    # Mask bearer tokens and headers
    text = _BEARER_RE.sub("Bearer [MASKED_TOKEN]", text)
    # Mask JWTs
    text = _JWT_RE.sub("[MASKED_TOKEN]", text)
    # Mask API keys
    text = _API_KEY_RE.sub("[MASKED_TOKEN]", text)
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


def format_sample_preview(val: Any, max_len: int = 80) -> str:
    """Formats a value into a safe, PII-masked, truncated string preview."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        masked = mask_pii(val)
        if len(masked) > max_len:
            return f"{masked[:max_len]}..."
        return masked
    if isinstance(val, dict):
        if not val:
            return "{}"
        try:
            preview_dict = {
                str(k): mask_pii(str(v)) if isinstance(v, str) else v
                for k, v in list(val.items())[:2]
            }
            s = json.dumps(preview_dict)
            if len(s) > max_len:
                return f"{s[:max_len]}..."
            return s
        except Exception:
            return "{...}"
    if isinstance(val, list):
        if not val:
            return "[]"
        sample_items: list[str] = []
        for x in val[:2]:
            if isinstance(x, str):
                sample_items.append(repr(mask_pii(x)))
            elif isinstance(x, bool):
                sample_items.append("true" if x else "false")
            elif isinstance(x, (int, float)):
                sample_items.append(str(x))
            elif x is None:
                sample_items.append("null")
            else:
                sample_items.append(format_sample_preview(x, max_len=20))
        if len(val) > len(sample_items):
            s = f"[{', '.join(sample_items)}, ...]"
        else:
            s = f"[{', '.join(sample_items)}]"
        if len(s) > max_len:
            return f"{s[:max_len]}..."
        return s
    return str(val)[:max_len]


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
            if not curr:
                if curr_path:
                    skeleton[curr_path] = {
                        "type": "dict",
                        "sample": "{}",
                    }
                return
            for key, val in curr.items():
                next_path = f"{curr_path}.{key}" if curr_path else str(key)
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
            close_idx = path.find("]", i)
            if close_idx == -1:
                return None
            content = path[i + 1:close_idx].strip()
            if not content:
                return None

            # Quoted key inside brackets: ['foo'] or ["foo"]
            if (content.startswith('"') and content.endswith('"')) or (
                content.startswith("'") and content.endswith("'")
            ):
                if len(content) < 2:
                    return None
                tokens.append(content[1:-1])
            elif content.isdigit():
                tokens.append(int(content))
            elif content.startswith("-") and content[1:].isdigit():
                tokens.append(int(content))
            elif content.isidentifier():
                tokens.append(content)
            else:
                return None
            i = close_idx + 1
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
