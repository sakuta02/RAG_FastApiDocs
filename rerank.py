"""Реранк кандидатов из гибридного поиска кросс-энкодером, чтобы топ получился точнее."""

import sys

from llama_index.core.postprocessor import SentenceTransformerRerank

from hybrid_search import CANDIDATE_POOL, get_hybrid_retriever
from search_index import get_index

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5

DEFAULT_PROMPT = "How do I add a dependency to a path operation?"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    retriever = get_candidate_retriever(get_index())
    reranker = get_reranker()

    print(f"Результаты после реранка для '{prompt}'")
    for rank, node in enumerate(rerank(prompt, retriever, reranker), start=1):
        print(f"\n#{rank}  Скор кросс-энкодера: {node.score:.4f}")
        print(node.metadata)
        print(node.text[:300])


def get_candidate_retriever(index):
    # на реранк отдаём CANDIDATE_POOL кандидатов, а до TOP_K сужает уже сам реранкер
    return get_hybrid_retriever(index, top_k=CANDIDATE_POOL)


def get_reranker():
    return SentenceTransformerRerank(model=RERANK_MODEL, top_n=TOP_K, device="mps")


def rerank(query, retriever, reranker):
    # кросс-энкодер читает вопрос и кандидата вместе, поэтому оценивает точнее dense-поиска
    return reranker.postprocess_nodes(retriever.retrieve(query), query_str=query)


if __name__ == "__main__":
    main()
