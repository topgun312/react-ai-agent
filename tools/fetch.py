"""fetch_pages — асинхронный параллельный HTTP-загрузка + извлечение текста из HTML.

Загружает несколько URL параллельно с помощью asyncio.gather().
Парсит HTML через BeautifulSoup, извлекая читаемый текстовый контент.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from bs4 import BeautifulSoup

from config.settings import settings

log = structlog.get_logger(__name__)

# Maximum content length per page (chars) to avoid filling context window.
# 5 pages × 3000 chars = ~15k chars ≈ 4k tokens — safe for most providers.
MAX_CONTENT_CHARS = 3000


async def fetch_pages(
    urls: list[str],
) -> list[dict[str, str]]:
    """Загрузить и извлечь текст из нескольких URL параллельно.

    Аргументы:
        urls: Список URL для загрузки. Все загружаются параллельно.

    Возвращает:
        Список словарей с ключами: url, title, content (или error при сбое).

    Raises:
        ToolError: Только если упали ВСЕ URL. Отдельные ошибки попадают
                   в результат как элементы с {"error": "..."}.

    Пример возврата:
        [
            {"url": "https://...", "title": "Название", "content": "...текст..."},
            {"url": "https://bad.url", "error": "Connection timeout"},
        ]
    """
    async with httpx.AsyncClient(
        timeout=settings.REQUEST_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "ResearchAgent/0.1 (educational bot)"},
    ) as client:
        raw_results = await asyncio.gather(
            *[_fetch_one(client, url) for url in urls],
            return_exceptions=True,
        )

    output: list[dict[str, str]] = []
    for url, result in zip(urls, raw_results):
        if isinstance(result, Exception):
            log.warning("fetch_failed", url=url, error=str(result))
            output.append({"url": url, "error": str(result)})
        else:
            output.append(result)

    success_count = sum(1 for r in output if "error" not in r)
    log.info("fetch_pages_done", total=len(urls), success=success_count)
    return output


async def _fetch_one(
    client: httpx.AsyncClient,
    url: str,
) -> dict[str, str]:
    """Загрузить один URL и извлечь текстовый контент через BeautifulSoup."""
    response = await client.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url
    # Remove noise elements
    for tag in soup(
        ["script", "style", "nav", "header", "footer", "aside", "noscript"]
    ):
        tag.decompose()

    # Extract main text
    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    text = "\n".join(lines)

    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + "\n... [truncated]"

    return {"url": url, "title": title, "content": text}
