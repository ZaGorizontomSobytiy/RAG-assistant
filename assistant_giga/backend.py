"""
Сборка pipeline для режима GigaChat.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List

from shared.cache import RAGCache
from shared.config import load_config
from shared.pipeline import RAGPipeline
from shared.vector_store import VectorStore

from assistant_giga.gigachat_client import GigaChatClient


def _root_and_config(config_path: str | Path | None) -> tuple[Path, Dict[str, Any]]:
    root = Path(__file__).resolve().parent.parent
    config = load_config(config_path or root / "config.yaml")
    return root, config


def create_embed_fn(root: Path, config: Dict[str, Any]) -> Callable[[str], List[float]]:
    client = GigaChatClient()

    def embed_fn(text: str) -> List[float]:
        return client.get_embeddings([text])[0]

    return embed_fn


def create_generate_fn(root: Path, config: Dict[str, Any]) -> Callable[[str, List], str]:
    client = GigaChatClient()
    model = config.get("models", {}).get("gigachat", "GigaChat")

    def generate_fn(prompt: str, context_docs: List) -> str:
        return client.chat_completion(
            messages=[
                {"role": "system", "content": "Ты — полезный AI-ассистент. Отвечай на основе контекста."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.3,
            max_tokens=500,
        ).strip()

    return generate_fn


def build_pipeline(
    data_path: str,
    reindex: bool = False,
    config_path: str | Path | None = None,
    verbose: bool = True,
) -> RAGPipeline:
    root, config = _root_and_config(config_path)
    chroma_cfg = config.get("chroma", {})
    chunk_cfg = config.get("chunking", {})
    search_cfg = config.get("search", {})
    cache_cfg = config.get("cache", {})
    models_cfg = config.get("models", {})

    persist = root / chroma_cfg.get("persist_directory", "chroma_db")
    persist.mkdir(parents=True, exist_ok=True)
    collection = chroma_cfg.get("collection_giga", "gigachat_rag_collection")
    cache_path = root / cache_cfg.get("giga_db_path", "gigachat_rag_cache.db")

    embed_fn = create_embed_fn(root, config)
    generate_fn = create_generate_fn(root, config)

    store = VectorStore(
        collection_name=collection,
        persist_directory=str(persist),
        embed_fn=embed_fn,
        chunk_size=chunk_cfg.get("chunk_size", 500),
        overlap=chunk_cfg.get("overlap", 100),
    )

    resolved_data = root / data_path if not os.path.isabs(data_path) else Path(data_path)
    if reindex or store.collection.count() == 0:
        store.index_path(str(resolved_data), reindex=reindex)

    cache = RAGCache(db_path=str(cache_path))
    model_name = models_cfg.get("gigachat", "GigaChat")

    return RAGPipeline(
        vector_store=store,
        cache=cache,
        generate_fn=generate_fn,
        model_name=model_name,
        top_k=search_cfg.get("top_k", 3),
        relevance_threshold=search_cfg.get("relevance_threshold"),
        verbose=verbose,
    )
