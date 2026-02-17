"""
Загрузка конфигурации из config.yaml.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Загрузить config.yaml из корня проекта.

    Args:
        config_path: путь к файлу; если None — ищется в корне (рядом с run.py).

    Returns:
        словарь конфигурации с ключами data_path, chroma, chunking, search, cache, models.
    """
    if config_path is None:
        root = Path(__file__).parent.parent
        config_path = root / "config.yaml"

    path = Path(config_path)
    if not path.exists():
        return _default_config()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if isinstance(data, dict) else _default_config()


def _default_config() -> Dict[str, Any]:
    return {
        "data_path": "data",
        "chroma": {
            "persist_directory": "chroma_db",
            "collection_api": "api_rag_collection",
            "collection_giga": "gigachat_rag_collection",
        },
        "chunking": {"chunk_size": 500, "overlap": 100},
        "search": {"top_k": 3, "relevance_threshold": 0.6},
        "cache": {
            "api_db_path": "api_rag_cache.db",
            "giga_db_path": "gigachat_rag_cache.db",
        },
        "models": {"openai": "gpt-4o-mini", "gigachat": "GigaChat"},
    }
