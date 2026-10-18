"""Гибридный поиск: dense (векторы) и BM25 (ключевые слова), результаты сливаем через RRF."""

import pickle
import sys

from build_bm25_index import BM25_INDEX_PATH, tokenize
from search_index import get_collection

RRF_K = 60
TOP_K = 5
CANDIDATE_POOL = 30  # сколько кандидатов берём из каждого поиска перед слиянием

DEFAULT_PROMPT = "How do I add a dependency to a path operation?"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    collection = get_collection()
    bm25_index = load_bm25_index()

    print(f"Гибридный поиск для '{prompt}'")
    results = hybrid_search(prompt, collection, bm25_index)
    for rank, (chunk_id, score) in enumerate(results, start=1):
        doc, metadata = lookup(chunk_id, bm25_index)
        print(f"\n#{rank}  RRF-скор: {score:.4f}")
        print(metadata)
        print(doc[:300])


def hybrid_search(query, collection, bm25_index, top_k=TOP_K):
    dense_ids = dense_search(query, collection)
    sparse_ids = sparse_search(query, bm25_index)
    fused = reciprocal_rank_fusion(dense_ids, sparse_ids)
    return fused[:top_k]


def dense_search(query, collection):
    result = collection.query(query_texts=[query], n_results=CANDIDATE_POOL)
    return result["ids"][0]


def sparse_search(query, bm25_index):
    scores = bm25_index["bm25"].get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [bm25_index["ids"][i] for i in ranked[:CANDIDATE_POOL]]


def reciprocal_rank_fusion(*ranked_lists):
    scores = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def load_bm25_index():
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def lookup(chunk_id, bm25_index):
    position = bm25_index["ids"].index(chunk_id)
    return bm25_index["documents"][position], bm25_index["metadatas"][position]


if __name__ == "__main__":
    main()
