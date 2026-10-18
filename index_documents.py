"""Режем документы и тикеты на чанки, эмбеддим их и сохраняем в Chroma."""

import json
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "doc_index"
PERSIST_DIR = "doc_index"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

DOCS_DIR = Path("source_documents")
TICKETS_FILE = Path("tickets/github_issues.jsonl")

HEADER_RE = re.compile(r"^#{1,6}\s")


def main():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    print("Режем документацию на чанки...")
    doc_chunks = chunk_docs(tokenizer)
    print("Режем тикеты на чанки...")
    ticket_chunks = chunk_tickets(tokenizer)
    all_chunks = doc_chunks + ticket_chunks
    print(f"Всего чанков: {len(all_chunks)} (из доков {len(doc_chunks)}, из тикетов {len(ticket_chunks)})")
    print("Эмбеддим и сохраняем в Chroma...")
    persist_chunks(all_chunks)
    print("Готово.")


def chunk_docs(tokenizer):
    chunks = []
    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        rel_path = str(md_file.relative_to(DOCS_DIR))
        page_url = "https://fastapi.tiangolo.com/" + rel_path.removesuffix(".md").removesuffix("/index") + "/"
        for heading, piece in split_markdown(text, tokenizer):
            chunks.append({
                "text": f"{heading}\n\n{piece}" if heading else piece,
                "metadata": {
                    "source_type": "docs",
                    "source_path": rel_path,
                    "heading": heading,
                    "url": page_url,
                },
            })
    return chunks


def chunk_tickets(tokenizer):
    chunks = []
    with open(TICKETS_FILE) as f:
        for line in f:
            ticket = json.loads(line)
            text = ticket_to_text(ticket)
            for piece in split_by_size(text, tokenizer):
                chunks.append({
                    "text": piece,
                    "metadata": {
                        "source_type": "ticket",
                        "source_path": f"issue #{ticket['number']}",
                        "heading": ticket["title"],
                        "url": ticket["url"],
                    },
                })
    return chunks


def ticket_to_text(ticket):
    parts = [ticket["title"], ticket["body"], *ticket["comments"]]
    return "\n\n".join(p for p in parts if p)


def split_markdown(text, tokenizer):
    result = []
    for heading, body in split_into_sections(text):
        for piece in split_by_size(body, tokenizer):
            if piece.strip():
                result.append((heading, piece.strip()))
    return result


def split_into_sections(text):
    # делим по строчкам с заголовками, но заголовки внутри ``` code fence
    # это не заголовки, а например "# (5)!" - аннотация в примере кода
    sections = []
    heading, lines, in_code_fence = "", [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code_fence = not in_code_fence
        if HEADER_RE.match(line) and not in_code_fence:
            sections.append((heading, "\n".join(lines)))
            heading, lines = line.strip(), []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))
    return sections


def split_by_size(text, tokenizer):
    """Упаковываем параграфы в чанки не больше CHUNK_SIZE токенов, а последние
    CHUNK_OVERLAP токенов каждого чанка переносим в следующий - это и есть overlap."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    pieces = []
    current, current_len = [], 0
    for para in paragraphs:
        para_len = token_len(para, tokenizer)
        if current and current_len + para_len > CHUNK_SIZE:
            pieces.append("\n\n".join(current))
            current = carry_overlap(current, tokenizer)
            current_len = sum(token_len(p, tokenizer) for p in current)
        current.append(para)
        current_len += para_len
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def carry_overlap(paragraphs, tokenizer):
    overlap, total = [], 0
    for para in reversed(paragraphs):
        para_len = token_len(para, tokenizer)
        if total + para_len > CHUNK_OVERLAP:
            break
        overlap.insert(0, para)
        total += para_len
    return overlap


def token_len(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def persist_chunks(chunks):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device="mps", normalize_embeddings=True
    )
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )


if __name__ == "__main__":
    main()
