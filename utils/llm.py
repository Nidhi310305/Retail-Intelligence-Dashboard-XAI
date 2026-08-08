from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv


load_dotenv()


def _get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@lru_cache(maxsize=1)
def get_gemini_client() -> Any:
    api_key = _get_api_key()
    if not api_key:
        return None

    try:
        from google import genai
    except Exception:
        return None

    return genai.Client(api_key=api_key)


def build_column_mapping_prompt(actual_columns: list[str], expected_roles: dict[str, str]) -> str:
    columns_json = json.dumps(actual_columns, ensure_ascii=True)
    roles_json = json.dumps(expected_roles, ensure_ascii=True)
    return f"""
You are mapping columns from a retail transactional dataset.

Actual columns:
{columns_json}

Expected roles and meanings:
{roles_json}

Return ONLY valid JSON with this shape:
{{
  "order_date": {{"column": "Order Date", "confidence": 0.98}},
  "ship_date": {{"column": "Ship Date", "confidence": 0.97}},
  "customer_id": {{"column": "Customer ID", "confidence": 0.99}},
  "sales": {{"column": "Sales", "confidence": 0.99}},
  "profit": {{"column": "Profit", "confidence": 0.99}},
  "discount": {{"column": "Discount", "confidence": 0.99}},
  "quantity": {{"column": "Quantity", "confidence": 0.99}},
  "region": {{"column": "Region", "confidence": 0.99}},
  "category": {{"column": "Category", "confidence": 0.99}}
}}

If a role has no good match, use null for column and confidence 0.
Do not include any explanation, markdown, or code fences.
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(cleaned[start : end + 1])


def generate_insights(context: str, question: str, system_prompt: str | None = None) -> str:
    client = get_gemini_client()
    if client is None:
        return "Insights are temporarily unavailable."

    prompt = system_prompt or "You are a retail business analyst."
    full_prompt = f"""
{prompt}

Context:
{context}

Question:
{question}

Answer in clear plain language and do not invent facts outside the context.
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=full_prompt,
        )
        return getattr(response, "text", "").strip() or "No response returned by Gemini."
    except Exception as exc:
        return "Insights are temporarily unavailable."


def parse_json_response(text: str) -> dict[str, Any]:
    return _extract_json(text)
