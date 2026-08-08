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
