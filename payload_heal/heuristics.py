"""Tier-1 Zero-LLM Deterministic Heuristic Matcher for payload-heal.

Recovers from schema drift (<1ms, $0 cost) by matching missing model fields to
candidate paths in the payload skeleton using:
- Case normalization (camelCase, PascalCase, kebab-case, snake_case)
- Prefix / suffix token analysis
- Nested leaf promotion (e.g., user_id -> data.user.id)
- Type compatibility filtering
- SequenceMatcher string similarity
- Ambiguity protection (preventing guesses when candidates are tied)
"""

from __future__ import annotations

import difflib
import re
from typing import Any

# ==============================================================================
# Key Normalization & Tokenization
# ==============================================================================

def tokenize_key(key: str) -> list[str]:
    """Splits a key into canonical lowercase constituent word tokens.

    Handles:
    - camelCase: `productPrice` -> `['product', 'price']`
    - PascalCase: `ProductPrice` -> `['product', 'price']`
    - kebab-case: `product-price` -> `['product', 'price']`
    - snake_case: `product_price` -> `['product', 'price']`
    - Acronyms: `productPriceUSD` -> `['product', 'price', 'usd']`
    - Numbers: `item1Price` -> `['item1', 'price']`
    """
    if not key or not isinstance(key, str):
        return []

    # Replace non-alphanumeric separators (dashes, dots, underscores, spaces)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", key)

    # Insert spaces between acronyms and following capitalized words: XMLParser -> XML Parser
    cleaned = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", cleaned)

    # Insert spaces between lowercase/digits and following uppercase: productPrice -> product Price
    cleaned = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cleaned)

    tokens = [t.lower() for t in cleaned.split() if t]
    return tokens


def normalize_key(key: str) -> str:
    """Converts camelCase, PascalCase, kebab-case, and snake_case into canonical snake_case.

    Example:
        `"productPriceUSD"` -> `"product_price_usd"`
        `"Product-Price"` -> `"product_price"`
    """
    tokens = tokenize_key(key)
    return "_".join(tokens)


# ==============================================================================
# Type Compatibility Checking
# ==============================================================================

def _parse_expected_types(annotation: str) -> set[str]:
    """Parses a Pydantic field annotation string into a set of normalized target type names.

    Handles:
    - Primitives: `float`, `int`, `str`, `bool`
    - Unions: `int | str`, `Union[float, int]`
    - Optionals: `Optional[float]`, `float | None`
    - Collections: `list[str]`, `dict[str, Any]`
    """
    if not annotation or not isinstance(annotation, str):
        return {"any"}

    cleaned = annotation.strip()
    # Strip builtins. or typing. prefixes
    cleaned = re.sub(r"\b(builtins|typing)\.", "", cleaned)

    types: set[str] = set()

    # Handle | (Python 3.10+ union syntax)
    if "|" in cleaned:
        parts = [p.strip() for p in cleaned.split("|")]
    elif cleaned.startswith("Union[") or cleaned.startswith("Optional["):
        inner = cleaned[cleaned.find("[") + 1 : cleaned.rfind("]")]
        parts = [p.strip() for p in inner.split(",")]
        if cleaned.startswith("Optional["):
            types.add("null")
    else:
        parts = [cleaned]

    for p in parts:
        p_lower = p.strip().lower()
        if p_lower in ("none", "nonetype", "null"):
            types.add("null")
        elif p_lower.startswith("list"):
            types.add("list")
        elif p_lower.startswith("dict"):
            types.add("dict")
        else:
            types.add(p_lower)

    return types


def is_type_compatible(
    expected_annotation: str,
    actual_type_name: str,
    sample_value: Any,
) -> bool:
    """Checks if a candidate JSON payload value can reasonably coerce into the
    expected Pydantic field type.

    Examples:
    - `float` accepts `int`, `float`, or numeric strings like `"129.99"`
    - `bool` accepts `bool`, `"true"`, `"false"`, `1`, `0`
    - `str` accepts `str`, `int`, `float`
    - `float` REJECTS non-numeric string titles like `"MacBook Pro"`
    """
    expected_types = _parse_expected_types(expected_annotation)

    if "any" in expected_types:
        return True

    # Check for null / None compatibility
    if actual_type_name in ("null", "nonetype"):
        return "null" in expected_types

    actual_lower = actual_type_name.lower()
    raw_sample_str = str(sample_value).strip() if sample_value is not None else ""

    for exp in expected_types:
        if exp == "float":
            if actual_lower in ("float", "int"):
                return True
            if actual_lower == "str" and raw_sample_str:
                try:
                    float(raw_sample_str)
                    return True
                except (ValueError, TypeError):
                    pass

        elif exp == "int":
            if actual_lower == "int":
                return True
            if actual_lower == "float":
                try:
                    val = float(raw_sample_str)
                    return val.is_integer()
                except (ValueError, TypeError):
                    pass
            if actual_lower == "str" and raw_sample_str:
                try:
                    int(raw_sample_str)
                    return True
                except (ValueError, TypeError):
                    pass

        elif exp == "bool":
            if actual_lower == "bool":
                return True
            if raw_sample_str.lower() in ("true", "false", "1", "0", "yes", "no"):
                return True
            if actual_lower == "int" and raw_sample_str in ("0", "1"):
                return True

        elif exp == "str":
            # Strings accept str, int, float, and bool
            if actual_lower in ("str", "int", "float", "bool"):
                return True

        elif exp == "list":
            if actual_lower.startswith("list"):
                return True

        elif exp == "dict":
            if actual_lower.startswith("dict"):
                return True

        elif exp == actual_lower:
            return True

    return False


# ==============================================================================
# Path Decomposition and Candidate Scoring
# ==============================================================================

def extract_path_segments(path: str) -> list[str]:
    """Extracts non-numeric key segments from a path.

    Example:
        `'data.items[0].pricing.priceUsd'` -> `['data', 'items', 'pricing', 'priceUsd']`
        `"data['price.usd']"` -> `['data', 'price.usd']`
    """
    if not path or not isinstance(path, str):
        return []

    from payload_heal.skeleton import parse_dot_path

    tokens = parse_dot_path(path)
    if tokens is not None:
        segments = [str(t) for t in tokens if isinstance(t, str)]
        if segments:
            return segments

    # Fallback regex parsing
    cleaned = re.sub(r"\[-?\d+\]", "", path)
    cleaned = re.sub(r"\[['\"](.*?)['\"]\]", r".\1", cleaned)

    segments = [s for s in cleaned.split(".") if s]
    return segments


def score_field_candidate(field_name: str, path: str) -> float:
    """Scores how well a candidate skeleton path matches a missing target field.

    Scoring incorporates:
    1. Exact normalized leaf match (e.g. `price_usd` vs `priceUsd` -> 0.95)
    2. Parent + leaf combination (e.g. `user_id` vs `user.id` -> 0.93)
    3. Prefix/suffix subset match (e.g. `title` vs `item_title` -> ~0.88)
    4. Token Jaccard overlap + difflib SequenceMatcher similarity
    """
    segments = extract_path_segments(path)
    if not segments:
        return 0.0

    leaf = segments[-1]
    parent = segments[-2] if len(segments) >= 2 else ""

    norm_target = normalize_key(field_name)
    target_tokens = tokenize_key(field_name)
    target_token_set = set(target_tokens)

    norm_leaf = normalize_key(leaf)
    leaf_tokens = tokenize_key(leaf)
    leaf_token_set = set(leaf_tokens)

    norm_parent = normalize_key(parent) if parent else ""
    parent_tokens = tokenize_key(parent) if parent else []
    parent_token_set = set(parent_tokens)

    # 1. Exact normalized leaf match
    if norm_target == norm_leaf:
        # If it's a direct top-level field match
        if len(segments) == 1:
            return 1.0
        return 0.95

    # 2. Parent + leaf combination equals target field name
    # e.g., target `user_id` matches path `data.user.id` where parent="user", leaf="id"
    combined_parent_leaf = f"{norm_parent}_{norm_leaf}" if norm_parent else norm_leaf
    combined_leaf_parent = f"{norm_leaf}_{norm_parent}" if norm_parent else norm_leaf

    if norm_target in (combined_parent_leaf, combined_leaf_parent):
        return 0.93

    if not target_token_set or not leaf_token_set:
        return 0.0

    # 3. Token overlap metrics
    common_tokens = target_token_set & leaf_token_set
    jaccard = len(common_tokens) / len(target_token_set | leaf_token_set)
    recall = len(common_tokens) / len(target_token_set)

    # SequenceMatcher similarity
    sm_leaf = difflib.SequenceMatcher(None, norm_target, norm_leaf).ratio()
    sm_combined = (
        difflib.SequenceMatcher(None, norm_target, combined_parent_leaf).ratio()
        if norm_parent
        else 0.0
    )
    sm_best = max(sm_leaf, sm_combined)

    # Full path token recall
    all_path_tokens = set(tokenize_key(path))
    path_recall = len(target_token_set & all_path_tokens) / len(target_token_set)

    # 4. Target is an exact subset of leaf tokens (prefix/suffix addition)
    # e.g. target="title", leaf="item_title"
    if target_token_set.issubset(leaf_token_set):
        score = 0.5 * recall + 0.3 * jaccard + 0.2 * sm_best + 0.10
        return min(0.86, round(score, 4))

    # Leaf is an exact subset of target tokens
    # e.g. target="item_title", leaf="title", parent="item"
    if leaf_token_set.issubset(target_token_set):
        if parent and parent_token_set & target_token_set:
            return 0.92
        score = 0.5 * jaccard + 0.5 * sm_best
        return round(score, 4)

    # 5. General weighted fuzzy combination
    score = 0.4 * jaccard + 0.3 * path_recall + 0.3 * sm_best
    return round(score, 4)


# ==============================================================================
# Heuristic Matcher Core
# ==============================================================================

def find_heuristic_matches(
    missing_fields: dict[str, str],
    skeleton: dict[str, dict[str, Any]],
    confidence_threshold: float = 0.78,
    margin: float = 0.05,
) -> dict[str, str]:
    """Finds deterministic heuristic matches for missing model fields.

    Parameters:
        missing_fields: Mapping of `field_name -> expected_type_str`.
        skeleton: Extracted payload skeleton mapping `dot_path -> {"type": ..., "sample": ...}`.
        confidence_threshold: Minimum confidence score required to accept a match (default: 0.78).
        margin: Minimum score difference between best and second-best candidate required
                to prevent ambiguous guesses (default: 0.05).

    Returns:
        Mapping of `field_name -> healed_source_path`.
    """
    if not missing_fields or not skeleton:
        return {}

    # Candidate evaluation per field: field_name -> [(score, path)]
    field_candidates: dict[str, list[tuple[float, str]]] = {}

    for field_name, expected_type in missing_fields.items():
        candidates: list[tuple[float, str]] = []

        for path, entry in skeleton.items():
            actual_type = entry.get("type", "")
            sample_val = entry.get("sample")

            # 1. Type compatibility filter
            if not is_type_compatible(expected_type, actual_type, sample_val):
                continue

            # 2. Compute similarity score
            score = score_field_candidate(field_name, path)
            if score >= confidence_threshold:
                candidates.append((score, path))

        # Sort candidate paths by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. Ambiguity protection: if multiple candidates are tied or nearly tied
        if candidates:
            best_score, best_path = candidates[0]
            if len(candidates) >= 2:
                second_best_score, _ = candidates[1]
                if (best_score - second_best_score) < margin:
                    # Ambiguous match! Reject to avoid incorrect guessing
                    continue
            field_candidates[field_name] = candidates

    # 4. Resolve 1-to-1 assignments across fields globally (highest confidence first)
    all_pairs: list[tuple[float, str, str]] = []
    for field_name, cand_list in field_candidates.items():
        for score, path in cand_list:
            all_pairs.append((score, field_name, path))

    all_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_paths: set[str] = set()
    assigned_fields: set[str] = set()
    matches: dict[str, str] = {}

    for score, field_name, path in all_pairs:
        if field_name not in assigned_fields and path not in assigned_paths:
            matches[field_name] = path
            assigned_fields.add(field_name)
            assigned_paths.add(path)

    return matches
