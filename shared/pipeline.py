"""
Общая логика RAG: кеш → поиск → промпт → LLM → кеш.
"""

from typing import Any, Callable, Dict, List

from shared.cache import RAGCache
from shared.vector_store import VectorStore

NO_CONTEXT_MESSAGE = (
    "В предоставленных документах нет информации для ответа на этот вопрос."
)


class RAGPipeline:
    """
    Универсальный RAG pipeline с инъекцией embed_fn и generate_fn.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        cache: RAGCache,
        generate_fn: Callable[[str, List[Dict[str, Any]]], str],
        model_name: str = "",
        top_k: int = 3,
        relevance_threshold: float | None = 0.6,
        verbose: bool = True,
    ):
        self.vector_store = vector_store
        self.cache = cache
        self.generate_fn = generate_fn
        self.model_name = model_name
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.verbose = verbose

    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        parts = [f"Документ {i}:\n{doc['text']}\n" for i, doc in enumerate(context_docs, 1)]
        context = "\n".join(parts)
        return f"""Ты — полезный AI-ассистент. Ответь на вопрос пользователя только на основе контекста ниже.

Контекст:
{context}

Вопрос: {query}

Инструкции:
- Отвечай только на основе контекста. Если в контексте нет информации для ответа — честно скажи об этом.
- Будь точным и кратким. Отвечай на русском языке.

Ответ:"""

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache:
            cached = self.cache.get(user_query)
            if cached:
                if self.verbose:
                    print("[+] Ответ из кеша")
                ctx = cached.get("context") or []
                context_docs = [{"text": t} for t in ctx] if ctx and isinstance(ctx[0], str) else ctx
                return {
                    "query": user_query,
                    "answer": cached["answer"],
                    "from_cache": True,
                    "context_docs": context_docs,
                    "cached_at": cached.get("created_at"),
                    "model": self.model_name,
                }

        context_docs = self.vector_store.search(
            user_query,
            top_k=self.top_k,
            relevance_threshold=self.relevance_threshold,
        )

        if not context_docs:
            if self.verbose:
                print("[*] Релевантный контекст не найден")
            return {
                "query": user_query,
                "answer": NO_CONTEXT_MESSAGE,
                "from_cache": False,
                "context_docs": [],
                "model": self.model_name,
            }

        prompt = self._build_prompt(user_query, context_docs)
        answer = self.generate_fn(prompt, context_docs)

        if use_cache:
            ctx_for_cache = [d["text"] for d in context_docs]
            self.cache.set(user_query, answer, ctx_for_cache)

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model_name,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model_name,
        }
