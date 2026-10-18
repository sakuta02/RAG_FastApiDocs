"""Режем документы и тикеты на ноды, эмбеддим их и сохраняем в Chroma через LlamaIndex."""

import json
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import AutoTokenizer

from search_index import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR, get_embed_model

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

DOCS_DIR = Path("source_documents")
TICKETS_FILE = Path("tickets/github_issues.jsonl")

# метаданные нужны для цитат и eval, но в эмбеддинг и в промпт им попадать не надо
HIDDEN_KEYS = ["source_type", "source_path", "url", "header_path"]


def main():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    # размер чанка меряем токенами эмбеддинг-модели, чтобы они не обрезались
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=lambda text: tokenizer.encode(text, add_special_tokens=False),
    )

    print("Режем документацию на ноды...")
    section_nodes = MarkdownNodeParser().get_nodes_from_documents(load_docs())
    doc_nodes = splitter(section_nodes)
    print("Режем тикеты на ноды...")
    ticket_nodes = splitter(load_tickets())
    nodes = doc_nodes + ticket_nodes
    print(f"Всего нод: {len(nodes)} (из доков {len(doc_nodes)}, из тикетов {len(ticket_nodes)})")

    print("Эмбеддим и сохраняем в Chroma...")
    persist_nodes(nodes)
    print("Готово.")


def load_docs():
    docs = []
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        rel_path = str(md_file.relative_to(DOCS_DIR))
        page_url = "https://fastapi.tiangolo.com/" + rel_path.removesuffix(".md").removesuffix("/index") + "/"
        docs.append(make_document(md_file.read_text(encoding="utf-8"), "docs", rel_path, page_url))
    return docs


def load_tickets():
    tickets = []
    with open(TICKETS_FILE) as f:
        for line in f:
            ticket = json.loads(line)
            parts = [ticket["title"], ticket["body"], *ticket["comments"]]
            text = "\n\n".join(p for p in parts if p)
            tickets.append(make_document(text, "ticket", f"issue #{ticket['number']}", ticket["url"]))
    return tickets


def make_document(text, source_type, source_path, url):
    return Document(
        text=text,
        metadata={"source_type": source_type, "source_path": source_path, "url": url},
        excluded_embed_metadata_keys=HIDDEN_KEYS,
        excluded_llm_metadata_keys=HIDDEN_KEYS,
    )


def persist_nodes(nodes):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=collection)
    )
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(),
        show_progress=True,
    )


if __name__ == "__main__":
    main()
