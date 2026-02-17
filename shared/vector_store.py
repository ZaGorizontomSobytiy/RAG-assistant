"""
Векторное хранилище на ChromaDB с метаданными и инкрементальной индексацией.
"""

import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import chromadb

from shared.chunking import chunk_text
from shared.document_loader import Document, load_from_path


def _sanitize_id(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)[:200]


class VectorStore:
    """
    ChromaDB-хранилище с метаданными (source, filename).
    Поддерживает полную и инкрементальную индексацию.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embed_fn: Callable[[str], List[float]],
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        self.collection_name = collection_name
        self.persist_directory = str(Path(persist_directory).resolve())
        self.embed_fn = embed_fn
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
            )
            print(f"Коллекция '{collection_name}' загружена. Документов: {self.collection.count()}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"Создана коллекция '{collection_name}'")

    def index_path(
        self,
        path: str,
        reindex: bool = False,
        recursive: bool = True,
    ) -> int:
        """
        Индексировать файл или директорию.

        Args:
            path: путь к файлу или папке
            reindex: если True — полная переиндексация (очистка и загрузка)
            recursive: обход подпапок для директории

        Returns:
            количество добавленных чанков
        """
        documents = load_from_path(path, recursive=recursive)
        if not documents:
            print("Нет документов для индексации.")
            return 0

        if reindex:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print("Коллекция очищена для полной переиндексации.")

        added = 0
        for doc in documents:
            added += self._index_document(doc)
        return added

    def _index_document(self, doc: Document) -> int:
        """Удалить старые чанки по source и добавить новые."""
        self.collection.delete(where={"source": doc.source})

        chunks = chunk_text(doc.text, self.chunk_size, self.overlap)
        if not chunks:
            return 0

        ids = []
        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = _sanitize_id(doc.source) + "_" + str(i)
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source": doc.source,
                "filename": doc.filename,
            })

        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self.embed_fn(text))
            if (i + 1) % 10 == 0:
                print(f"  Обработано {i + 1}/{len(texts)} чанков")

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  Добавлено {len(chunks)} чанков из {doc.filename}")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        relevance_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Поиск релевантных чанков.

        Args:
            query: запрос
            top_k: число результатов
            relevance_threshold: макс. cosine distance; если лучший результат хуже — вернуть пусто
            where: фильтр по метаданным ChromaDB

        Returns:
            список {id, text, distance, source?, filename?}
        """
        query_embedding = self.embed_fn(query)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "distances", "metadatas"],
        }
        if where is not None:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        if not results["documents"] or not results["documents"][0]:
            return []

        out = []
        for i in range(len(results["documents"][0])):
            dist = results["distances"][0][i]
            if relevance_threshold is not None and dist > relevance_threshold:
                continue
            meta = (results["metadatas"][0][i] or {}) if results["metadatas"] else {}
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": dist,
                "source": meta.get("source"),
                "filename": meta.get("filename"),
            })

        return out

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }
