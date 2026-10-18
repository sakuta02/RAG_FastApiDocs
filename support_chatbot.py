"""Весь RAG целиком: поиск, реранк, ответ Qwen и эскалация на человека,
если контекста не хватает, вместо того чтобы гадать."""

import json
import re
import sys

import ollama
from sentence_transformers import CrossEncoder

import hybrid_search
import rerank as rerank_module
from search_index import get_collection

LLM_MODEL = "qwen2.5:3b-instruct"
TOP_K = 5
MIN_RERANK_SCORE = 0.0

DEFAULT_PROMPT = "How do I add a dependency to a path operation?"

CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """You are a support assistant for the FastAPI framework.
Answer ONLY using the numbered context passages below. Do not use any
outside knowledge, even if you happen to know the answer. Write a real,
complete answer in your own words, and cite the passages you used inline,
like [1] or [2].

Reply with a single JSON object with exactly two fields:
- "answer": your answer text, with inline [n] citations
- "confidence": "high" if the context fully answers the question, "low" if
  the context is missing, irrelevant, or only partially covers the question

If the context doesn't answer the question, set confidence to "low" and say
in "answer" that you don't have enough information - never guess."""


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    collection = get_collection()
    bm25_index = hybrid_search.load_bm25_index()
    cross_encoder = CrossEncoder(rerank_module.RERANK_MODEL, device="mps")

    result = answer_question(prompt, collection, bm25_index, cross_encoder)
    print_result(result)


def answer_question(question, collection, bm25_index, cross_encoder):
    results = rerank_module.rerank(question, collection, bm25_index, cross_encoder, top_k=TOP_K)
    context, sources = build_context(results)
    reply = ask_llm(question, context)

    best_score = results[0][2]
    escalate = bool(reply["confidence"] == "low" or best_score < MIN_RERANK_SCORE)
    cited = sorted({int(n) for n in CITATION_RE.findall(reply["answer"])})
    used = [sources[i - 1] for i in cited if 1 <= i <= len(sources)]
    return {
        "question": question,
        "answer": reply["answer"],
        "confidence": reply["confidence"],
        "escalate": escalate,
        "sources": used,
        "failed": reply.get("failed", False),
    }


def build_context(results):
    lines, sources = [], []
    for i, (doc, metadata, _) in enumerate(results, start=1):
        lines.append(f"[{i}] (from {metadata['source_path']})\n{doc}")
        sources.append(metadata["url"])
    return "\n\n".join(lines), sources


def ask_llm(question, context):
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            format="json",
            # без этого метрики плавают от запуска к запуску
            options={"temperature": 0, "seed": 0},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"},
            ],
        )
        reply = json.loads(response.message.content)
    except (ollama.ResponseError, json.JSONDecodeError):
        # на temperature=0 модель иногда зацикливается на повторе токенов и
        # Ollama обрывает запрос; без ответа остаётся только эскалировать
        return {"answer": "", "confidence": "low", "failed": True}
    # 3B модель иногда отдаёт answer списком предложений вместо строки
    reply["answer"] = coerce_text(reply.get("answer", ""))
    reply["confidence"] = reply.get("confidence", "low")
    return reply


def coerce_text(value):
    if isinstance(value, list):
        return " ".join(coerce_text(v) for v in value)
    return str(value)


def print_result(result):
    if result["escalate"]:
        print("[Эскалация на человека - бот не уверен в ответе]")
    print(f"\nВопрос: {result['question']}")
    print(f"Ответ: {result['answer']}")
    print(f"Уверенность: {result['confidence']}")
    if result["sources"]:
        print("Источники:")
        for url in result["sources"]:
            print(f"  - {url}")


if __name__ == "__main__":
    main()
