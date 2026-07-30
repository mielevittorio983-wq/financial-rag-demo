from __future__ import annotations

import math
from collections import Counter
from typing import Any

from reg_rag.retrieval.simple import SearchHit, record_text, tokenize


def vector_search(records: list[dict[str, Any]], query: str, top_k: int = 8) -> list[SearchHit]:
    query_terms = Counter(tokenize(query))
    if not query_terms or not records:
        return []
    document_terms = [Counter(tokenize(record_text(record))) for record in records]
    document_frequency = Counter(term for terms in document_terms for term in terms)
    count = len(records)
    query_vector = _tfidf(query_terms, document_frequency, count)
    hits = []
    for record, terms in zip(records, document_terms):
        score = _cosine(query_vector, _tfidf(terms, document_frequency, count))
        if score > 0:
            hits.append(SearchHit(score=score, record=record))
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:top_k]


def _tfidf(terms: Counter[str], document_frequency: Counter[str], document_count: int) -> dict[str, float]:
    total = sum(terms.values()) or 1
    return {
        term: (frequency / total) * (math.log((document_count + 1) / (document_frequency[term] + 1)) + 1)
        for term, frequency in terms.items()
    }


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    shared = left.keys() & right.keys()
    numerator = sum(left[term] * right[term] for term in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
