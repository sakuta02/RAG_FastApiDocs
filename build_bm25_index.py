"""Строим BM25-индекс (поиск по ключевым словам) по тем же чанкам, что лежат в Chroma."""

import pickle
import re

from chromadb import PersistentClient

from rank_bm25 import BM25Okapi

COLLECTION_NAME = "doc_index"
PERSIST_DIR = "doc_index"
BM25_INDEX_PATH = "bm25_index.pkl"

TOKEN_RE = re.compile(r"\w+")


def main():
    print("Читаем чанки из Chroma...")
    ids, documents, metadatas = load_chunks()
    print(f"Токенизируем {len(documents)} чанков...")
    tokenized = [tokenize(doc) for doc in documents]
    print("Строим BM25-индекс...")
    bm25 = BM25Okapi(tokenized)
    print("Сохраняем индекс...")
    save_index(bm25, ids, documents, metadatas)
    print("Готово.")


def load_chunks():
    client = PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get()
    return result["ids"], result["documents"], result["metadatas"]


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def save_index(bm25, ids, documents, metadatas):
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "ids": ids, "documents": documents, "metadatas": metadatas}, f
        )


if __name__ == "__main__":
    main()
