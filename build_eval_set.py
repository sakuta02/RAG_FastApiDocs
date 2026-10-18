"""Собираем eval-набор: вопросы с известным источником и вопросы-ловушки,
ответов на которые в корпусе нет."""

import json
import random

import ollama

from search_index import get_vector_store

LLM_MODEL = "qwen2.5:3b-instruct"
SAMPLE_SIZE = 85
MIN_CHUNK_CHARS = 300
OUTPUT_FILE = "eval_dataset.jsonl"
SEED = 42

QUESTION_PROMPT = """Here is a passage from the FastAPI documentation:

---
{chunk}
---

Write ONE natural question that a developer would ask, whose answer is fully
contained in this passage. Reply with only the question, nothing else."""


TRAP_QUESTIONS = [
    "How do I configure Redis Sentinel failover in a Django app?",
    "What's the best way to set up Kubernetes horizontal pod autoscaling for a Spring Boot service?",
    "How do I train a custom tokenizer for a BERT model from scratch?",
    "What's the capital of Mongolia?",
    "How do I configure AWS Lambda cold start optimization for a Go binary?",
    "How does the Rust borrow checker handle mutable references in closures?",
    "How do I tune autovacuum settings for a large PostgreSQL table?",
    "How do I share state between components with React hooks?",
    "How do I configure remote state locking in Terraform with S3?",
    "How do Kafka consumer groups rebalance partitions?",
    "How do I use VLOOKUP across two Excel sheets?",
    "How do I undo a git rebase that already got pushed?",
    "How does NumPy broadcasting work for arrays with different shapes?",
    "How do I center a div with CSS flexbox?",
    "What is the difference between a mutex and a semaphore?",
]


def main():
    random.seed(SEED)
    print("Загружаем чанки документации...")
    chunks = load_doc_chunks()
    sample = random.sample(chunks, SAMPLE_SIZE)

    print(f"Генерируем {SAMPLE_SIZE} вопросов через {LLM_MODEL}...")
    entries = [build_entry(chunk) for chunk in sample]

    print(f"Добавляем {len(TRAP_QUESTIONS)} вопросов-ловушек...")
    entries += [trap_entry(q) for q in TRAP_QUESTIONS]

    print(f"Сохраняем {len(entries)} вопросов в {OUTPUT_FILE}...")
    save_entries(entries)
    print("Готово.")


def load_doc_chunks():
    nodes = get_vector_store().get_nodes(None)
    chunks = [(n.text, n.metadata) for n in nodes if n.metadata["source_type"] == "docs"]
    return [c for c in chunks if len(c[0]) > MIN_CHUNK_CHARS]


def build_entry(chunk):
    text, metadata = chunk
    question = ask_llm_for_question(text)
    return {
        "question": question,
        "source_path": metadata["source_path"],
        "heading": text.split("\n", 1)[0],
        "answerable": True,
    }


def ask_llm_for_question(chunk_text):
    response = ollama.chat(
        model=LLM_MODEL,
        options={"temperature": 0, "seed": 0},
        messages=[{"role": "user", "content": QUESTION_PROMPT.format(chunk=chunk_text)}],
    )
    return response.message.content.strip()


def trap_entry(question):
    return {
        "question": question,
        "source_path": None,
        "heading": None,
        "answerable": False,
    }


def save_entries(entries):
    with open(OUTPUT_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
