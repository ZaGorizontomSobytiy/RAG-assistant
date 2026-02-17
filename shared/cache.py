"""
Кеш пар вопрос–ответ для RAG (SQLite).
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional


class RAGCache:
    """Кеш результатов RAG-запросов."""

    def __init__(self, db_path: str = "rag_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def _get_query_hash(self, query: str) -> str:
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        h = self._get_query_hash(query)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT query, answer, context, created_at FROM cache WHERE query_hash = ?",
            (h,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "query": row[0],
            "answer": row[1],
            "context": json.loads(row[2]) if row[2] else None,
            "created_at": row[3],
            "from_cache": True,
        }

    def set(
        self,
        query: str,
        answer: str,
        context: Optional[list] = None,
    ) -> None:
        h = self._get_query_hash(query)
        ctx_json = json.dumps(context) if context else None
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at) VALUES (?, ?, ?, ?, ?)",
            (h, query, answer, ctx_json, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def clear(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        dates = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM cache").fetchone()
        conn.close()
        size = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
        return {
            "total_entries": count,
            "oldest_entry": dates[0] if dates and dates[0] else None,
            "newest_entry": dates[1] if dates and dates[1] else None,
            "db_size_mb": size,
        }
