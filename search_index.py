"""Самый простой скрипт для поиска по эмбеддингам (только dense-поиск, без изысков)."""

import sys

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "doc_index"
PERSIST_DIR = "li_index"
DEFAULT_PROMPT = "How do I add a dependency to a path operation?"


def main():
    retriever = get_index().as_retriever(similarity_top_k=5)

    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT

    print(f"Ищем подходящие куски документов для '{prompt}'")
    for node in retriever.retrieve(prompt):
        print(f"\nСходство (чем больше, тем лучше): {node.score:.4f}")
        print(node.metadata)
        print(node.text[:300])


def get_embed_model():
    # та же модель и нормализация, что и при создании индекса
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL, device="mps", normalize=True, embed_batch_size=64
    )


def get_vector_store():
    collection = chromadb.PersistentClient(path=PERSIST_DIR).get_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


def get_index():
    return VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=get_embed_model())


if __name__ == "__main__":
    main()
