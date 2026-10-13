"""summarize_page — LLM-сжатие текста.

Сжимает длинный контент страницы до ключевых моментов, релевантных
исследовательскому запросу. Вызывается, когда контент страницы слишком
длинный для прямой передачи в основной контекст LLM.
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.llm_client import create_llm_client
from config.settings import settings
from tools.registry import ToolError

log = structlog.get_logger(__name__)

# Клиент кэшируется: один и тот же провайдер/модель переиспользуются между вызовами
_cached_llm: Any = None
_cache_key: tuple[str, str] | None = None


def _get_llm() -> Any:
    """Вернуть LLM-клиент для текущего провайдера из настроек."""
    global _cached_llm, _cache_key
    key = (settings.LLM_PROVIDER, settings.DEFAULT_MODEL)
    if _cached_llm is None or _cache_key != key:
        _cached_llm = create_llm_client()
        _cache_key = key
    return _cached_llm


SUMMARIZE_PROMPT = """\
Summarize the following content, focusing on: {focus}

Extract the most important facts, statistics, and insights.
Be concise — aim for 200-400 words.

Content:
{content}
"""


async def summarize_page(
    content: str,
    focus: str = "key findings and insights",
) -> str:
    """Сжать контент с помощью LLM.

    Аргументы:
        content: Полный текстовый контент для сжатия.
        focus: Какой аспект должен быть в центре внимания резюме.

    Возвращает:
        Краткое резюме (200–400 слов).

    Raises:
        ToolError: При ошибке API LLM.
    """
    try:
        prompt = SUMMARIZE_PROMPT.format(focus=focus, content=content[:12000])
        response = await _get_llm().complete(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="",
        )

        text_parts = [
            block["text"]
            for block in response.get("content", [])
            if block.get("type") == "text"
        ]
        summary = "\n".join(text_parts).strip()
        if not summary:
            raise ToolError("LLM returned empty summary", tool_name="summarize_page")

        log.info(
            "summarize_done",
            provider=settings.LLM_PROVIDER,
            model=settings.DEFAULT_MODEL,
            chars_in=len(content),
            chars_out=len(summary),
        )
        return summary

    except ToolError:
        raise
    except Exception as e:
        raise ToolError(str(e), tool_name="summarize_page") from e
