"""Весь RAG целиком: поиск, реранк, ответ Qwen и эскалация на человека,
если контекста не хватает, вместо того чтобы гадать."""

import re
import sys
from typing import Literal

import ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel

from rerank import get_candidate_retriever, get_reranker, rerank
from search_index import get_index

LLM_MODEL = "qwen2.5:3b-instruct"
MIN_RERANK_SCORE = 0.0

DEFAULT_PROMPT = "How do I add a dependency to a path operation?"

CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """You are a support assistant for the FastAPI framework.
Answer ONLY using the numbered context passages below. Do not use any
outside knowledge, even if you happen to know the answer. Write a real,
complete answer in your own words, and cite the passages you used inline,
like [1] or [2].

Fill in two fields:
- "answer": your answer text, with inline [n] citations
- "confidence": "high" if the context fully answers the question, "low" if
  the context is missing, irrelevant, or only partially covers the question

If the context doesn't answer the question, set confidence to "low" and say
in "answer" that you don't have enough information - never guess."""

PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content="Context:\n\n{context}\n\nQuestion: {question}"),
    ]
)


class Reply(BaseModel):
    answer: str
    confidence: Literal["high", "low"]


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    index = get_index()
    retriever = get_candidate_retriever(index)

    result = answer_question(prompt, retriever, get_reranker(), get_llm())
    print_result(result)


def get_llm():
    # temperature=0 и seed=0, иначе метрики плавают от запуска к запуску
    return Ollama(
        model=LLM_MODEL,
        temperature=0,
        request_timeout=120.0,
        additional_kwargs={"seed": 0},
    )


def answer_question(question, retriever, reranker, llm):
    nodes = rerank(question, retriever, reranker)
    context, sources = build_context(nodes)
    reply = ask_llm(llm, question, context)

    best_score = nodes[0].score
    escalate = bool(reply is None or reply.confidence == "low" or best_score < MIN_RERANK_SCORE)
    answer = reply.answer if reply else ""
    cited = sorted({int(n) for n in CITATION_RE.findall(answer)})
    used = [sources[i - 1] for i in cited if 1 <= i <= len(sources)]
    return {
        "question": question,
        "answer": answer,
        "confidence": reply.confidence if reply else "low",
        "escalate": escalate,
        "sources": used,
        "failed": reply is None,
    }


def build_context(nodes):
    lines, sources = [], []
    for i, node in enumerate(nodes, start=1):
        lines.append(f"[{i}] (from {node.metadata['source_path']})\n{node.text}")
        sources.append(node.metadata["url"])
    return "\n\n".join(lines), sources


def ask_llm(llm, question, context):
    # схема уходит в Ollama как format, поэтому ответ не может выйти за её рамки
    try:
        return llm.structured_predict(Reply, PROMPT, context=context, question=question)
    except (ollama.ResponseError, ValueError):
        # на temperature=0 модель иногда зацикливается на повторе токенов и
        # Ollama обрывает запрос; без ответа остаётся только эскалировать
        return None


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
