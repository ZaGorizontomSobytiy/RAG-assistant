"""
Единая точка входа: индексация и интерактивный чат RAG-ассистента.

Использование:
  python run.py --api                    # чат с OpenAI
  python run.py --giga                   # чат с GigaChat
  python run.py --api --data ./docs      # данные из ./docs
  python run.py --giga --data ./docs --reindex   # полная переиндексация
  python run.py --api --data ./docs --index-only # только индексация, без чата
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env из корня проекта
ROOT = Path(__file__).resolve().parent
env_file = ROOT / ".env"
if not env_file.exists():
    env_file = ROOT / "env.example"
load_dotenv(env_file)


def _print_banner(mode: str) -> None:
    title = "RAG Ассистент (GigaChat)" if mode == "giga" else "RAG Ассистент (OpenAI API)"
    print(f"\n{'='*60}\n  {title}\n{'='*60}")
    print("  exit / quit — выход; stats — статистика; clear — очистка кеша\n")


def _print_response(result: dict) -> None:
    print(f"\n{'-'*60}\nВопрос: {result['query']}\n{'-'*60}")
    if result.get("from_cache"):
        print("Источник: КЕШ")
        if result.get("cached_at"):
            print(f"   Сохранено: {result['cached_at']}")
    else:
        print(f"Модель: {result.get('model', '')}")
        print(f"   Документов в контексте: {len(result.get('context_docs', []))}")
    print(f"\nОтвет:\n{result['answer']}")
    if not result.get("from_cache") and result.get("context_docs"):
        print("\nКонтекст (фрагмент):")
        for i, doc in enumerate(result["context_docs"][:2], 1):
            preview = doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
            print(f"   {i}. {preview}")
    print(f"{'-'*60}\n")


def _print_stats(pipeline) -> None:
    stats = pipeline.get_stats()
    print(f"\n{'='*60}\nСТАТИСТИКА\n{'='*60}")
    print(f"  Векторное хранилище: {stats['vector_store']['name']}, записей: {stats['vector_store']['count']}")
    print(f"  Кеш: записей {stats['cache']['total_entries']}, размер {stats['cache']['db_size_mb']:.2f} MB")
    print(f"  Модель: {stats['model']}\n{'='*60}\n")


def run_index(mode: str, data_path: str, reindex: bool, config_path: str | None) -> None:
    from shared.config import load_config
    from shared.vector_store import VectorStore

    config = load_config(config_path or ROOT / "config.yaml")
    chroma_cfg = config.get("chroma", {})
    chunk_cfg = config.get("chunking", {})

    persist = ROOT / chroma_cfg.get("persist_directory", "chroma_db")
    persist.mkdir(parents=True, exist_ok=True)
    collection = (
        chroma_cfg.get("collection_giga", "gigachat_rag_collection")
        if mode == "giga"
        else chroma_cfg.get("collection_api", "api_rag_collection")
    )

    if mode == "api":
        from assistant_api.backend import create_embed_fn
    else:
        from assistant_giga.backend import create_embed_fn

    embed_fn = create_embed_fn(ROOT, config)
    store = VectorStore(
        collection_name=collection,
        persist_directory=str(persist),
        embed_fn=embed_fn,
        chunk_size=chunk_cfg.get("chunk_size", 500),
        overlap=chunk_cfg.get("overlap", 100),
    )

    resolved = ROOT / data_path if not os.path.isabs(data_path) else Path(data_path)
    n = store.index_path(str(resolved), reindex=reindex)
    print(f"\n[OK] Индексация завершена. Добавлено чанков: {n}\n")


def run_chat(mode: str, data_path: str, reindex: bool, config_path: str | None) -> None:
    if mode == "api":
        if not os.getenv("OPENAI_API_KEY"):
            print("[!] OPENAI_API_KEY не установлен. Задайте в .env или окружении.")
            sys.exit(1)
        from assistant_api.backend import build_pipeline
    else:
        if not os.getenv("GIGACHAT_AUTH_KEY") or not os.getenv("GIGACHAT_RQUID"):
            print("[!] GIGACHAT_AUTH_KEY и GIGACHAT_RQUID должны быть в .env")
            sys.exit(1)
        from assistant_giga.backend import build_pipeline

    pipeline = build_pipeline(
        data_path=data_path,
        reindex=reindex,
        config_path=config_path or ROOT / "config.yaml",
        verbose=True,
    )
    _print_banner(mode)

    while True:
        try:
            user_input = input("Ваш вопрос: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nДо свидания!\n")
                break
            if user_input.lower() == "stats":
                _print_stats(pipeline)
                continue
            if user_input.lower() == "clear":
                confirm = input("Очистить кеш? (yes/no): ")
                if confirm.lower() in ("yes", "y", "да"):
                    pipeline.cache.clear()
                    print("[OK] Кеш очищен\n")
                continue
            if not user_input:
                continue
            result = pipeline.query(user_input)
            _print_response(result)
        except KeyboardInterrupt:
            print("\n\nДо свидания!\n")
            break
        except Exception as e:
            print(f"\n[!] Ошибка: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-ассистент: OpenAI или GigaChat")
    parser.add_argument("--api", action="store_true", help="Режим OpenAI API")
    parser.add_argument("--giga", action="store_true", help="Режим GigaChat")
    parser.add_argument("--data", type=str, default=None, help="Путь к файлу или папке с документами")
    parser.add_argument("--reindex", action="store_true", help="Полная переиндексация перед запуском")
    parser.add_argument("--index-only", action="store_true", help="Только индексация, без чата")
    parser.add_argument("--config", type=str, default=None, help="Путь к config.yaml")
    args = parser.parse_args()

    if not args.api and not args.giga:
        parser.error("Укажите --api или --giga")

    mode = "api" if args.api else "giga"

    from shared.config import load_config
    config = load_config(args.config or ROOT / "config.yaml")
    data_path = args.data or config.get("data_path", "data")

    if args.index_only:
        run_index(mode, data_path, args.reindex, args.config)
        return

    run_chat(mode, data_path, args.reindex, args.config)


if __name__ == "__main__":
    main()
