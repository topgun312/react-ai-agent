"""write_report — терминальный инструмент: собирает итоговый исследовательский отчёт.

Это условие остановки для ReAct-цикла.
Когда оркестратор видит write_report в блоке tool_use, он должен:
1. Вызвать эту функцию для форматирования отчёта
2. Записать результат в state.report
3. Немедленно выйти из цикла

Не вызывать этот инструмент несколько раз за одну сессию.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


@dataclass
class ReportResult:
    """Структурированный результат работы write_report."""

    title: str
    content: str
    sources: list[dict[str, str]]
    word_count: int


async def write_report(
    title: str,
    content: str,
    sources: list[dict[str, str]] | None = None,
) -> ReportResult:
    """Отформатировать и вернуть итоговый исследовательский отчёт.

    Аргументы:
        title: Название отчёта.
        content: Полный Markdown-контент с разделами и строчными ссылками.
        sources: Список {"url": ..., "title": ...} для секции References.

    Возвращает:
        ReportResult с отформатированным Markdown-контентом.

    Примечание:
        Эта функция форматирует отчёт — она НЕ вызывает LLM.
        LLM уже написала контент в вызове tool_use.
        Здесь мы только собираем итоговую структуру документа.
    """
    sources = sources or []
    full_content = content

    if sources:
        refs = "\n\n## References\n\n"
        for i, src in enumerate(sources, 1):
            src_title = src.get("title") or src.get("url", "")
            src_url = src.get("url", "")
            refs += f"{i}. [{src_title}]({src_url})\n"
        full_content = content.rstrip() + refs

    word_count = len(full_content.split())

    log.info("report_written", title=title, word_count=word_count, sources=len(sources))
    return ReportResult(
        title=title, content=full_content, sources=sources, word_count=word_count
    )
