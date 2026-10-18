"""Гибридный поиск: dense (векторы) и BM25 (ключевые слова), результаты сливаем через RRF."""

import sys

from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from build_bm25_index import BM25_INDEX_PATH
from search_index import get_index

TOP_K = 5
CANDIDATE_POOL = 30  # сколько кандидатов берём из каждого поиска перед слиянием

DEFAULT_PROMPT = "How do I add a dependency to a path operation?"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    retriever = get_hybrid_retriever(get_index())

    print(f"Гибридный поиск для '{prompt}'")
    for rank, node in enumerate(retriever.retrieve(prompt), start=1):
        print(f"\n#{rank}  RRF-скор: {node.score:.4f}")
        print(node.metadata)
        print(node.text[:300])


def get_hybrid_retriever(index, top_k=TOP_K):
    vector_retriever = index.as_retriever(similarity_top_k=CANDIDATE_POOL)
    bm25_retriever = BM25Retriever.from_persist_dir(BM25_INDEX_PATH)
    bm25_retriever.similarity_top_k = CANDIDATE_POOL
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        # подзапросы не генерируем (num_queries=1), поэтому настоящая LLM тут не нужна
        llm=MockLLM(),
        mode="reciprocal_rerank",
        similarity_top_k=top_k,
        num_queries=1,
        use_async=False,
    )


if __name__ == "__main__":
    main()
