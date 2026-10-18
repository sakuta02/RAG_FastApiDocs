"""Самый простой скрипт для поиска по эмбеддингам (только dense-поиск, без изысков)."""

import sys

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "doc_index"
PERSIST_DIR = "doc_index"
DEFAULT_PROMPT = "How do I add a dependency to a path operation?"


def main():
    collection = get_collection()

    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT

    print(f"Ищем подходящие куски документов для '{prompt}'")
    results = collection.query(query_texts=[prompt], n_results=5)
    for doc, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"\nКосинусное расстояние (чем меньше, тем лучше): {distance:.4f}")
        print(metadata)
        print(doc[:300])


def get_collection():
    # берём ту же модель и ту же метрику расстояния, что и при создании индекса
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="mps", normalize_embeddings=True
    )
    client = PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)


if __name__ == "__main__":
    main()
