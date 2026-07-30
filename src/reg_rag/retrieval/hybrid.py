from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from reg_rag.retrieval.simple import SearchHit, record_text, search_records, tokenize
from reg_rag.retrieval.vector import vector_search


def hybrid_search(records: list[dict[str, Any]], query: str, top_k: int = 8, pool_size: int = 50) -> list[SearchHit]:
    lexical_hits = search_records(records, query, top_k=pool_size)
    bm25_hits = _bm25_search(records, query, top_k=pool_size)
    vector_hits = vector_search(records, query, top_k=pool_size)
    fused = _rrf_fuse([lexical_hits, bm25_hits, vector_hits])
    return fused[:top_k]


def _bm25_search(records: list[dict[str, Any]], query: str, top_k: int) -> list[SearchHit]:
    corpus = [tokenize(record_text(record)) for record in records]
    query_tokens = tokenize(query)
    if not query_tokens or not corpus:
        return []
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    query_set = set(query_tokens)
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (
            float(item[1]) + (len(query_set & set(corpus[item[0]])) / max(len(query_set), 1) * 5.0)
        ),
        reverse=True,
    )
    hits: list[SearchHit] = []
    for index, score in ranked[:top_k]:
        if not set(corpus[index]) & set(query_tokens):
            continue
        overlap_score = len(query_set & set(corpus[index])) / max(len(query_set), 1)
        hits.append(SearchHit(score=float(score) + overlap_score * 5.0, record=records[index]))
    return hits


def bm25_search(records: list[dict[str, Any]], query: str, top_k: int = 8) -> list[SearchHit]:
    return _bm25_search(records, query, top_k=top_k)


def _rrf_fuse(hit_lists: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    by_key: dict[str, tuple[float, SearchHit]] = {}
    for list_index, hits in enumerate(hit_lists):
        for rank, hit in enumerate(hits, start=1):
            key = _record_key(hit.record)
            weight = 2.0 if list_index == 0 else 1.0
            contribution = weight / (k + rank)
            existing = by_key.get(key)
            if existing:
                score, best_hit = existing
                by_key[key] = (score + contribution, best_hit if best_hit.score >= hit.score else hit)
            else:
                by_key[key] = (contribution, hit)

    fused: list[SearchHit] = []
    for score, hit in by_key.values():
        fused.append(SearchHit(score=score, record=hit.record))
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("chunk_id") or record.get("cell_id") or id(record))
