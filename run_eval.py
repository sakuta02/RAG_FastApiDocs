"""Считаем Recall@5 и MRR ретривера по eval_dataset.jsonl.

Запуск: python run_eval.py [dense|hybrid|rerank]
"""

import json
import sys

from sentence_transformers import CrossEncoder

import hybrid_search
import rerank as rerank_module
from search_index import get_collection

EVAL_FILE = "eval_dataset.jsonl"
TOP_K = 5


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dense"
    search_fn = build_search_fn(mode)
    eval_set = [entry for entry in load_eval_set() if entry["answerable"]]

    hits, reciprocal_ranks = 0, []
    for entry in eval_set:
        rank = find_rank(search_fn(entry["question"]), entry["source_path"])
        reciprocal_ranks.append(1 / rank if rank else 0)
        if rank:
            hits += 1

    total = len(eval_set)
    print(f"Ретривер: {mode}")
    print(f"Отвечаемых вопросов: {total}")
    print(f"Recall@{TOP_K}: {hits}/{total} = {hits / total:.1%}")
    print(f"MRR: {sum(reciprocal_ranks) / total:.3f}")


def build_search_fn(mode):
    collection = get_collection()
    if mode == "dense":
        return lambda question: dense_paths(collection, question)
    if mode == "hybrid":
        bm25_index = hybrid_search.load_bm25_index()
        return lambda question: hybrid_paths(collection, bm25_index, question)
    if mode == "rerank":
        bm25_index = hybrid_search.load_bm25_index()
        cross_encoder = CrossEncoder(rerank_module.RERANK_MODEL, device="mps")
        return lambda question: rerank_paths(collection, bm25_index, cross_encoder, question)
    raise ValueError(f"Unknown retriever mode: {mode}")


def dense_paths(collection, question):
    results = collection.query(query_texts=[question], n_results=TOP_K)
    return [m["source_path"] for m in results["metadatas"][0]]


def hybrid_paths(collection, bm25_index, question):
    fused = hybrid_search.hybrid_search(question, collection, bm25_index, top_k=TOP_K)
    return [hybrid_search.lookup(chunk_id, bm25_index)[1]["source_path"] for chunk_id, _ in fused]


def rerank_paths(collection, bm25_index, cross_encoder, question):
    results = rerank_module.rerank(question, collection, bm25_index, cross_encoder, top_k=TOP_K)
    return [metadata["source_path"] for _, metadata, _ in results]


def load_eval_set():
    with open(EVAL_FILE) as f:
        return [json.loads(line) for line in f]


def find_rank(paths, correct_path):
    if correct_path in paths:
        return paths.index(correct_path) + 1
    return None


if __name__ == "__main__":
    main()
