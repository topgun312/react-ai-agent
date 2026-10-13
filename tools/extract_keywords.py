"""extract_keywords — локальное извлечение ключевых слов из текста.

Оценивает частотность слов в тексте и возвращает самые частые (без LLM).
Используется для выделения главных тем перед написанием отчёта.
"""

from __future__ import annotations

import re
from collections import Counter

import structlog

from tools.registry import ToolError

log = structlog.get_logger(__name__)


async def extract_keywords(
    text: str,
    max_keywords: int = 10,
) -> list[str]:
    """Извлечь ключевые слова из текста по частоте.

    Args:
        text: Текст для анализа.
        max_keywords: Количество ключевых слов (1-20). Default: 10.

    Returns:
        Список ключевых слов по убыванию частоты.
    """
    if not text or len(text.strip()) < 10:
        raise ToolError("Text is too short", tool_name="extract_keywords")

    # Только слова длиннее 3 символов, без стоп-слов
    words = re.findall(r"\b[a-zA-Zа-яА-Я]{4,}\b", text.lower())
    stop_words = {"this", "that", "with", "from", "have", "been", "they"}
    filtered = [w for w in words if w not in stop_words]

    if not filtered:
        raise ToolError("No keywords found", tool_name="extract_keywords")

    keywords = [word for word, _ in Counter(filtered).most_common(max_keywords)]
    log.info("keywords_extracted", count=len(keywords))
    return keywords
