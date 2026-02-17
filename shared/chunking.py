"""
Разбиение текста на чанки с учётом абзацев и предложений.
"""

import re
from typing import List


def _get_overlap_text(text: str, overlap_size: int) -> str:
    if len(text) <= overlap_size:
        return text
    overlap_candidate = text[-overlap_size:]
    for delim in [". ", "! ", "? ", "\n"]:
        pos = overlap_candidate.find(delim)
        if pos != -1:
            return overlap_candidate[pos + len(delim) :].strip()
    return overlap_candidate.strip()


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
    sentences = re.split(r"([.!?]+\s+)", paragraph)
    full_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            full_sentences.append(sentences[i] + sentences[i + 1])
        else:
            full_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        full_sentences.append(sentences[-1])

    chunks = []
    current_chunk = ""
    for sentence in full_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk = (current_chunk + " " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
                overlap_text = _get_overlap_text(current_chunk, overlap)
                current_chunk = (overlap_text + " " + sentence) if overlap_text else sentence
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk_len: int = 50,
) -> List[str]:
    """
    Разбиение текста на чанки: приоритет абзацам, затем по предложениям.

    Returns:
        список чанков (отфильтрованы слишком короткие).
    """
    paragraphs = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + paragraph) if current_chunk else paragraph
        elif current_chunk:
            chunks.append(current_chunk)
            overlap_text = _get_overlap_text(current_chunk, overlap)
            current_chunk = (overlap_text + "\n\n" + paragraph) if overlap_text else paragraph
        else:
            if len(paragraph) > chunk_size:
                sentence_chunks = _split_long_paragraph(paragraph, chunk_size, overlap)
                if sentence_chunks:
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if len(c) >= min_chunk_len]
