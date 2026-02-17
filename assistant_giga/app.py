"""
Точка входа для RAG-ассистента (GigaChat).
Рекомендуется запуск из корня: python run.py --giga
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from dotenv import load_dotenv

load_dotenv(root / ".env")
load_dotenv(root / "env.example")

if __name__ == "__main__":
    from run import run_chat
    run_chat("giga", "data", False, None)
