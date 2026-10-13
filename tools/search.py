"""search_web — веб-поиск через DuckDuckGo (без API-ключа, бесплатно).

Использует библиотеку duckduckgo-search. DDGS.text() — синхронный,
поэтому запускается в пуле потоков через asyncio.to_thread().

Результат нормализован к формату {url, title, snippet} —
такому же, как ожидает оркестратор.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from tools.registry import ToolError

log = structlog.get_logger(__name__)


def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Синхронный вызов DuckDuckGo — выполняется в отдельном потоке."""
    from ddgs import DDGS  # noqa: PLC0415
    from ddgs.exceptions import (  # noqa: PLC0415
        DDGSException,
        RatelimitException,
        TimeoutException,
    )

    try:
        raw: list[dict[str, Any]] = DDGS().text(query, max_results=max_results) or []
    except (RatelimitException, TimeoutException, DDGSException) as e:
        raise ToolError(str(e), tool_name="search_web") from e

    return [
        {
            "url": r.get("href", ""),
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw
        if r.get("href")
    ]


async def search_web(
    query: str,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Поиск в интернете через DuckDuckGo.

    Args:
        query: Поисковый запрос.
        max_results: Максимальное количество результатов (1–10).

    Returns:
        Список словарей: url, title, snippet.

    Raises:
        ToolError: При сбое поиска (rate limit, timeout, ошибка сети).

    Example return:
        [
            {"url": "https://...", "title": "...", "snippet": "..."},
            ...
        ]
    """
    try:
        results = await asyncio.to_thread(_ddg_search, query, max_results)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(str(e), tool_name="search_web") from e

    log.info("search_web_done", query=query, count=len(results))
    return results
