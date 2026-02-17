# RAG-ассистент

Консольный RAG (Retrieval-Augmented Generation) с поддержкой **OpenAI API** и **GigaChat**. Индексация документов из нескольких форматов, кеш ответов, опциональная оценка качества через RAGAS.

## Основные возможности

- **Два режима:** OpenAI API и GigaChat — один интерфейс, переключение флагом.
- **Форматы документов:** .txt, .md, .pdf, .docx; можно указать файл или папку (рекурсивный обход).
- **Векторное хранилище:** ChromaDB, метаданные (источник, имя файла), семантический поиск.
- **Инкрементальная индексация:** добавление и обновление документов без полной переиндексации; опция полной переиндексации (`--reindex`).
- **Режим «только индексация»:** `--index-only` для скриптов и cron без запуска чата.
- **Кеш:** SQLite-кеш пар вопрос–ответ для экономии запросов к API.
- **Конфиг:** `config.yaml` — пути к данным, параметры chunking, поиска и моделей.
- **Оценка качества:** скрипт оценки RAG через RAGAS (режим OpenAI).
- **Python 3.11**, зависимости в `requirements.txt`.

## Требования

- **Python 3.11.9** (рекомендуется)
- Виртуальное окружение (рекомендуется)

## Установка

### 1. Python 3.11

- **Windows (winget):** `winget install Python.Python.3.11`
- Создание venv: `py -3.11 -m venv .venv`
- Активация (Windows): `.venv\Scripts\activate`
- Активация (Linux/macOS): `source .venv/bin/activate`

### 2. Зависимости

Из корня проекта:

```bash
pip install -r requirements.txt
```

### 3. Переменные окружения

Скопируйте шаблон и заполните значения (в репозиторий не коммитить):

```bash
# Windows (PowerShell)
copy env.example .env

# Linux/macOS
cp env.example .env
```

В `.env` укажите:

- **Режим OpenAI (`--api`):** `OPENAI_API_KEY=...`
- **Режим GigaChat (`--giga`):** `GIGACHAT_AUTH_KEY=...`, `GIGACHAT_RQUID=...`

## Запуск

Все команды выполняются **из корня проекта**.

### Интерактивный чат

```bash
# OpenAI API
python run.py --api

# GigaChat
python run.py --giga
```

### Данные

По умолчанию используются документы из папки `data` (см. `config.yaml` → `data_path`). Своя папка или файл:

```bash
python run.py --api --data ./docs
python run.py --giga --data ./my_file.pdf
```

Поддерживаемые форматы: **.txt**, **.md**, **.pdf**, **.docx**. Для директории выполняется обход (в т.ч. вложенных папок).

### Переиндексация и только индексация

```bash
# Полная переиндексация, затем чат
python run.py --api --data ./docs --reindex

# Только проиндексировать (без чата), удобно для cron/скриптов
python run.py --api --data ./docs --index-only
python run.py --giga --data ./docs --reindex --index-only
```

### Оценка качества (RAGAS, только для API)

```bash
python -m assistant_api.evaluate_ragas
```

Требуется `OPENAI_API_KEY`. Запуск из корня проекта.

## Конфигурация

Файл **config.yaml** в корне:

- `data_path` — путь к данным по умолчанию
- `chroma` — директория ChromaDB и имена коллекций
- `chunking` — размер чанка и перекрытие
- `search` — число документов в контексте (`top_k`), порог релевантности
- `cache` — пути к SQLite-кешу
- `models` — имена моделей OpenAI и GigaChat

Переменные CLI (`--data`, `--config`) переопределяют настройки из конфига.

## Структура проекта

```
├── run.py                 # Точка входа (CLI)
├── config.yaml            # Конфигурация
├── env.example            # Шаблон .env
├── requirements.txt
├── data/                  # Документы по умолчанию
│   └── docs.txt
├── shared/                # Общая логика
│   ├── config.py
│   ├── document_loader.py # Загрузка .txt, .md, .pdf, .docx
│   ├── chunking.py
│   ├── cache.py
│   ├── vector_store.py    # ChromaDB, метаданные, инкремент
│   └── pipeline.py        # RAG: кеш → поиск → LLM
├── assistant_api/         # Режим OpenAI
│   ├── backend.py         # Сборка pipeline
│   ├── app.py             # Обёртка для python -m assistant_api.app
│   └── evaluate_ragas.py  # Оценка RAGAS
└── assistant_giga/        # Режим GigaChat
    ├── backend.py
    ├── gigachat_client.py
    └── app.py
```

## Команды в чате

- `exit` / `quit` / `q` — выход
- `stats` — статистика (хранилище, кеш, модель)
- `clear` — очистка кеша (с подтверждением)

## Безопасность

- Не коммитить `.env` и реальные ключи (в `.gitignore` уже учтено).
- Использовать `env.example` только как шаблон без значений.
