"""Считаем Recall@5 и MRR ретривера по eval_dataset.jsonl.

Запуск: python run_eval.py [dense|hybrid|rerank]
"""

import json
import sys

from hybrid_search import get_hybrid_retriever
from rerank import get_candidate_retriever, get_reranker, rerank
from search_index import get_index

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
    index = get_index()
    if mode == "dense":
        retriever = index.as_retriever(similarity_top_k=TOP_K)
        return lambda question: retriever.retrieve(question)
    if mode == "hybrid":
        retriever = get_hybrid_retriever(index, top_k=TOP_K)
        return lambda question: retriever.retrieve(question)
    if mode == "rerank":
        retriever, reranker = get_candidate_retriever(index), get_reranker()
        return lambda question: rerank(question, retriever, reranker)
    raise ValueError(f"Unknown retriever mode: {mode}")


def load_eval_set():
    with open(EVAL_FILE) as f:
        return [json.loads(line) for line in f]


def find_rank(nodes, correct_path):
    paths = [node.metadata["source_path"] for node in nodes]
    if correct_path in paths:
        return paths.index(correct_path) + 1
    return None


if __name__ == "__main__":
    main()
