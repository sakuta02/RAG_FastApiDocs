"""Реранк кандидатов из гибридного поиска кросс-энкодером, чтобы топ получился точнее."""

import sys

from sentence_transformers import CrossEncoder

import hybrid_search
from search_index import get_collection

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5
CANDIDATE_POOL = 30  # сколько результатов гибридного поиска реранкаем перед отсечением до TOP_K

DEFAULT_PROMPT = "How do I add a dependency to a path operation?"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    collection = get_collection()
    bm25_index = hybrid_search.load_bm25_index()
    cross_encoder = CrossEncoder(RERANK_MODEL, device="mps")

    print(f"Результаты после реранка для '{prompt}'")
    results = rerank(prompt, collection, bm25_index, cross_encoder)
    for rank, (doc, metadata, score) in enumerate(results, start=1):
        print(f"\n#{rank}  Скор кросс-энкодера: {score:.4f}")
        print(metadata)
        print(doc[:300])


def rerank(query, collection, bm25_index, cross_encoder, top_k=TOP_K):
    candidates = hybrid_search.hybrid_search(query, collection, bm25_index, top_k=CANDIDATE_POOL)
    docs_and_meta = [hybrid_search.lookup(chunk_id, bm25_index) for chunk_id, _ in candidates]

    # кросс-энкодер читает вопрос и кандидата вместе, поэтому оценивает точнее dense-поиска
    scores = cross_encoder.predict([(query, doc) for doc, _ in docs_and_meta])

    reranked = sorted(zip(docs_and_meta, scores), key=lambda pair: pair[1], reverse=True)
    return [(doc, meta, score) for (doc, meta), score in reranked[:top_k]]


if __name__ == "__main__":
    main()
