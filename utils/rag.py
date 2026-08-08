from __future__ import annotations

from typing import Any

import numpy as np
import traceback

from .llm import generate_insights, get_gemini_client


def build_knowledge_base(*args: Any, **kwargs: Any) -> list[str]:
    if args:
        findings = args[0]
        if isinstance(findings, list):
            return findings
    knowledge_base = kwargs.get("knowledge_base")
    if isinstance(knowledge_base, list):
        return knowledge_base
    return []


def build_dashboard_knowledge_base(*, summary_chunks: list[str] | None = None, region_sales: Any = None, monthly_sales: Any = None, rfm_frame: Any = None, rfm_summary: dict[str, Any] | None = None, anomaly_artifacts: Any = None, forecast_artifacts: Any = None) -> list[str]:
    chunks: list[str] = []
    if summary_chunks:
        chunks.extend(chunk for chunk in summary_chunks if chunk)

    if region_sales is not None and not getattr(region_sales, "empty", True):
        try:
            region_column = region_sales.columns[0]
            sales_column = region_sales.columns[1]
            top_rows = region_sales.sort_values(sales_column, ascending=False).head(3)
            ranked = ", ".join(f"{row[region_column]} (${row[sales_column]:,.0f})" for _, row in top_rows.iterrows())
            chunks.append(f"Sales by Region: {ranked}.")
        except Exception:
            pass

    if monthly_sales is not None and not getattr(monthly_sales, "empty", True):
        try:
            date_column = monthly_sales.columns[0]
            sales_column = monthly_sales.columns[1]
            ordered = monthly_sales.sort_values(date_column)
            first_row = ordered.iloc[0]
            last_row = ordered.iloc[-1]
            peak_row = ordered.sort_values(sales_column, ascending=False).iloc[0]
            chunks.append(
                f"Sales Trend: from {first_row[date_column]} sales were ${first_row[sales_column]:,.0f}; the latest period {last_row[date_column]} was ${last_row[sales_column]:,.0f}; peak period was {peak_row[date_column]} at ${peak_row[sales_column]:,.0f}."
            )
        except Exception:
            pass

    if rfm_frame is not None and not getattr(rfm_frame, "empty", True):
        try:
            if "Segment" in rfm_frame.columns:
                segment_counts = rfm_frame["Segment"].value_counts().to_dict()
                parts = ", ".join(f"{segment}: {count}" for segment, count in segment_counts.items())
                chunks.append(f"Customer Segments: {parts}.")
                if {"Recency", "Frequency", "Monetary"}.issubset(rfm_frame.columns):
                    cluster_summary = rfm_frame.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1).to_dict("index")
                    chunks.append(f"RFM Summary: {cluster_summary}.")
        except Exception:
            pass

    if rfm_summary:
        chunks.append(f"RFM Model Summary: {rfm_summary}.")

    if anomaly_artifacts is not None:
        try:
            count = getattr(anomaly_artifacts, "count", None)
            pct = getattr(anomaly_artifacts, "pct", None)
            top_features = getattr(anomaly_artifacts, "top_features", None)
            chunks.append(f"Anomaly Detection: {count} transactions ({pct}%) were flagged as unusual. Top contributing features: {top_features}.")
            frame = getattr(anomaly_artifacts, "frame", None)
            if frame is not None and not getattr(frame, "empty", True) and "Anomaly" in frame.columns:
                anomalies = frame[frame["Anomaly"] == -1].copy()
                if not anomalies.empty:
                    anomaly_columns = [column for column in ["Order ID", "Sales", "Quantity", "Discount", "Profit"] if column in anomalies.columns]
                    if anomaly_columns:
                        sample = anomalies[anomaly_columns].head(3).to_dict("records")
                        chunks.append(f"Anomaly Examples: {sample}.")
        except Exception:
            pass

    if forecast_artifacts is not None:
        try:
            metrics = getattr(forecast_artifacts, "metrics", {})
            top_feature = getattr(forecast_artifacts, "top_feature", None)
            chunks.append(f"Forecast Summary: model metrics {metrics}; top driver {top_feature}.")
        except Exception:
            pass

    return chunks


def _embed_text(text: str) -> np.ndarray | None:
    client = get_gemini_client()
    if client is None:
        return None
    try:
        result = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return np.array(result.embeddings[0].values)
    except Exception:
        return None


def _retrieve_with_embeddings(query: str, knowledge_base: list[str], top_k: int = 3) -> list[str]:
    query_embedding = _embed_text(query)
    if query_embedding is None or not knowledge_base:
        lowered_query = query.lower()
        scores = []
        for chunk in knowledge_base:
            score = sum(1 for token in lowered_query.split() if token in chunk.lower())
            scores.append(score)
        ranked = [chunk for _, chunk in sorted(zip(scores, knowledge_base), key=lambda item: item[0], reverse=True)]
        return ranked[:top_k]

    chunk_embeddings = []
    for chunk in knowledge_base:
        embedding = _embed_text(chunk)
        if embedding is None:
            return knowledge_base[:top_k]
        chunk_embeddings.append(embedding)

    matrix = np.array(chunk_embeddings)
    similarities = np.dot(matrix, query_embedding) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_embedding)
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [knowledge_base[index] for index in top_indices]


def retrieve_context(query: str, knowledge_base: list[str], top_k: int = 3) -> list[str]:
    return _retrieve_with_embeddings(query, knowledge_base, top_k=top_k)


def answer_question(query: str, knowledge_base: list[str], top_k: int = 3) -> str:
    context_chunks = retrieve_context(query, knowledge_base, top_k=top_k)
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)
    prompt = f"""
You are a retail business analyst answering questions only from the provided context.
If the context does not contain enough information, say that honestly.

Context:
{context}

Question: {query}

Answer in 2-4 sentences. Do not invent facts.
""".strip()

    client = get_gemini_client()
    if client is None:
        return "Answers are temporarily unavailable right now."

    try:
        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        return getattr(response, "text", "").strip() or "No response returned by Gemini."
    except Exception as exc:
        print(f"[Gemini] RAG generate_content failed: {exc}")
        print(traceback.format_exc())
        return "Answers are temporarily unavailable right now."


def summarize_findings(query: str, knowledge_base: list[str]) -> str:
    context = "\n".join(retrieve_context(query, knowledge_base))
    return generate_insights(context, query)
