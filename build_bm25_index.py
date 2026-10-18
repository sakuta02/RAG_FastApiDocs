"""Строим BM25-индекс (поиск по ключевым словам) по тем же нодам, что лежат в Chroma."""

from llama_index.retrievers.bm25 import BM25Retriever

from search_index import get_vector_store

BM25_INDEX_PATH = "li_bm25"


def main():
    print("Читаем ноды из Chroma...")
    # берём ноды из того же хранилища, чтобы фьюжн потом узнавал дубликаты по хэшу
    nodes = get_vector_store().get_nodes(None)
    print(f"Строим BM25-индекс по {len(nodes)} нодам...")
    retriever = BM25Retriever.from_defaults(nodes=nodes)
    print("Сохраняем индекс...")
    retriever.persist(BM25_INDEX_PATH)
    print("Готово.")


if __name__ == "__main__":
    main()
