from typing import Any

from reg_rag.retrieval.hybrid import bm25_search, hybrid_search
from reg_rag.retrieval.query_expansion import expand_query
from reg_rag.retrieval.rerank import RerankerName, rerank_hits
from reg_rag.retrieval.simple import SearchHit, search_records
from reg_rag.retrieval.vector import vector_search


def retrieve_records(
    records: list[dict[str, Any]],
    query: str,
    top_k: int = 8,
    mode: str = "simple",
    reranker: RerankerName = "none",
    pool_size: int = 50,
) -> list[SearchHit]:
    search_functions = {
        "simple": search_records,
        "bm25": bm25_search,
        "vector": vector_search,
        "hybrid": hybrid_search,
    }
    if mode not in search_functions:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    search_fn = search_functions[mode]
    expanded_query = expand_query(query)
    hits = search_fn(records, expanded_query, top_k=max(top_k, pool_size) if reranker != "none" else top_k)
    return rerank_hits(hits, expanded_query, reranker=reranker, top_k=top_k)


__all__ = [
    "RerankerName",
    "SearchHit",
    "bm25_search",
    "expand_query",
    "hybrid_search",
    "retrieve_records",
    "rerank_hits",
    "search_records",
    "vector_search",
]
