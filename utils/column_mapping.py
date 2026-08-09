from __future__ import annotations

from typing import Any
import concurrent.futures
import difflib
import traceback

from .llm import build_column_mapping_prompt, get_gemini_client, parse_json_response


EXPECTED_ROLES: dict[str, str] = {
    "order_date": "Date the order was placed",
    "ship_date": "Date the order was shipped",
    "customer_id": "Unique customer identifier or name",
    "sales": "Revenue amount for the transaction",
    "profit": "Profit amount for the transaction",
    "discount": "Discount applied, as a decimal or percentage",
    "quantity": "Number of units ordered",
    "region": "Geographic region of the order",
    "category": "Product category",
}

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "order_date": ("order date", "ordered date", "date ordered"),
    "ship_date": ("ship date", "shipped date"),
    "customer_id": ("customer id", "customer name", "customer", "client"),
    "sales": ("sales", "revenue", "amount", "total sales", "order value"),
    "profit": ("profit", "margin", "net profit"),
    "discount": ("discount", "discount rate", "promo discount"),
    "quantity": ("quantity", "units", "qty"),
    "region": ("region", "area", "territory"),
    "category": ("category", "product category", "product type"),
}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _heuristic_match(columns: list[str], role: str) -> dict[str, Any]:
    normalized_columns = {_normalize(column): column for column in columns}
    aliases = ROLE_ALIASES.get(role, ())

    for alias in aliases:
        if alias in normalized_columns:
            return {"column": normalized_columns[alias], "confidence": 0.95, "source": "heuristic"}

    for column in columns:
        normalized_column = _normalize(column)
        if any(alias in normalized_column or normalized_column in alias for alias in aliases):
            confidence = 0.9 if role not in {"customer_id", "ship_date"} else 0.85
            return {"column": column, "confidence": confidence, "source": "heuristic"}

    return {"column": None, "confidence": 0.0, "source": "heuristic"}


def infer_column_mapping(columns: list[str]) -> dict[str, Any]:
    """Infer expected role-to-column mappings using Gemini with confidence handling."""
    mapping: dict[str, Any] = {}
    for role in EXPECTED_ROLES:
        mapping[role] = _heuristic_match(columns, role)

    client = get_gemini_client()
    if client is None:
        return {"mapping": mapping, "unresolved_roles": [role for role, info in mapping.items() if not info.get("column") or info.get("confidence", 0) < 0.8], "raw": None}

    prompt = build_column_mapping_prompt(columns, EXPECTED_ROLES)
    parsed: dict[str, Any] = {}
    LLM_TIMEOUT = 15
    try:
        def _call_generate() -> str:
            resp = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
            return getattr(resp, "text", "") or ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call_generate)
            try:
                text = fut.result(timeout=LLM_TIMEOUT)
                if text:
                    parsed = parse_json_response(text)
            except concurrent.futures.TimeoutError:
                print(f"[Gemini] generate_content timed out after {LLM_TIMEOUT} seconds")
                parsed = {}
    except Exception as exc:
        print(f"[Gemini] generate_content failed: {exc}")
        traceback.print_exc()
        parsed = {}

    for role in EXPECTED_ROLES:
        candidate = parsed.get(role)
        if isinstance(candidate, dict):
            column = candidate.get("column")
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            if isinstance(column, str) and column in columns:
                mapping[role] = {"column": column, "confidence": confidence, "source": "llm"}
            elif column is None:
                mapping[role] = {"column": None, "confidence": confidence, "source": "llm"}

    # If the LLM failed / timed out or didn't provide useful suggestions, try a fuzzy matching
    unresolved = [role for role, info in mapping.items() if not info.get("column") or float(info.get("confidence", 0.0)) < 0.8]
    if not parsed or unresolved:
        normalized_columns = {_normalize(c): c for c in columns}

        def _fuzzy_for_role(role_name: str) -> dict[str, Any]:
            # consider expected role name and aliases as candidates
            candidates = [role_name.replace("_", " ")] + list(ROLE_ALIASES.get(role_name, ()))
            best = None
            best_score = 0.0
            for cand in candidates:
                # match against normalized columns
                matches = difflib.get_close_matches(_normalize(cand), list(normalized_columns.keys()), n=1, cutoff=0.6)
                if matches:
                    score = difflib.SequenceMatcher(a=_normalize(cand), b=matches[0]).ratio()
                    if score > best_score:
                        best_score = score
                        best = normalized_columns[matches[0]]
            if best:
                return {"column": best, "confidence": round(0.6 + best_score * 0.35, 2), "source": "fuzzy"}
            return {"column": None, "confidence": 0.0, "source": "fuzzy"}

        for role in unresolved:
            # only replace if still unresolved or low confidence
            if not mapping.get(role) or float(mapping.get(role, {}).get("confidence", 0.0)) < 0.8:
                mapping[role] = _fuzzy_for_role(role)

    unresolved_roles = [role for role, info in mapping.items() if not info.get("column") or float(info.get("confidence", 0.0)) < 0.8]
    return {"mapping": mapping, "unresolved_roles": unresolved_roles, "raw": parsed}


def resolve_required_roles(mapping: dict[str, Any]) -> dict[str, str]:
    """Extract resolved required roles from a richer mapping payload."""
    resolved: dict[str, str] = {}
    source_mapping = mapping.get("mapping", mapping)
    for role, info in source_mapping.items():
        if isinstance(info, dict) and info.get("column"):
            resolved[role] = str(info["column"])
        elif isinstance(info, str):
            resolved[role] = info
    return resolved
