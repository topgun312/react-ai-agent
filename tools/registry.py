"""ToolRegistry — центральный реестр всех инструментов агента.

Обязанности:
- Хранить схемы инструментов (формат JSON Schema для API Anthropic)
- Диспатчить вызовы инструментов по имени на нужную async-функцию
- Проверять имена инструментов перед диспатчем

Все новые инструменты ОБЯЗАНЫ быть зарегистрированы здесь (схема + диспатч).
Оркестратор никогда не вызывает инструменты напрямую — всегда через реестр.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ToolError(Exception):
    """Поднимается инструментами при устранимых сбоях.

    Оркестратор ловит это исключение и добавляет его как результат tool,
    позволяя LLM понять причину сбоя и попробовать альтернативы.
    """

    def __init__(self, message: str, tool_name: str = "") -> None:
        """Инициализировать исключение.

        Аргументы:
            message:   Текст ошибки, передаваемый LLM как tool result.
            tool_name: Имя инструмента, вызвавшего ошибку.
        """
        super().__init__(message)
        self.tool_name = tool_name


# ── Tool schemas (JSON Schema for Anthropic API) ──────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "search_web",
        "description": (
            "Search the internet for information on a topic. "
            "Returns a list of relevant URLs with titles and snippets. "
            "Use this first to discover sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string. Be specific and include key terms.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10). Default: 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_pages",
        "description": (
            "Fetch and extract text content from one or more URLs in parallel. "
            "Returns cleaned text content for each page. "
            "Use after search_web to get full page content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of URLs to fetch. Will be fetched concurrently.",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "summarize_page",
        "description": (
            "Summarize a long text into key points relevant to the research query. "
            "Use this when page content is too long to include directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The full text content to summarize.",
                },
                "focus": {
                    "type": "string",
                    "description": "What aspect to focus on when summarizing.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "write_report",
        "description": (
            "Synthesize all research findings into a final structured Markdown report. "
            "CALL THIS when you have gathered enough information. "
            "This terminates the research session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Report title.",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Full Markdown report with sections, findings, and citations. "
                        "Include numbered references at the end."
                    ),
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                        },
                        "required": ["url", "title"],
                    },
                    "description": "List of sources cited in the report.",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "extract_keywords",
        "description": (
            "Extract key terms and concepts from text content. "
            "Use this to identify main topics before writing the report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text content to analyze.",
                },
                "max_keywords": {
                    "type": "integer",
                    "description": "Max keywords to return (1-20). Default: 10.",
                    "default": 10,
                },
            },
            "required": ["text"],
        },
    },
]

# ── Tool dispatch table ───────────────────────────────────────────────────────

ToolFn = Callable[..., Awaitable[Any]]

# Populated at the bottom of this file after importing tool modules.
TOOL_DISPATCH: dict[str, ToolFn] = {}


# ── ToolRegistry ──────────────────────────────────────────────────────────────

# ── Argument name aliases ─────────────────────────────────────────────────────
# Open-source LLMs often use different argument names than the schema specifies.
# Map known aliases → canonical names per tool.

_ARG_ALIASES: dict[str, dict[str, str]] = {
    "fetch_pages": {
        "url_list": "urls",
        "url": "urls",
        "page_urls": "urls",
        "pages": "urls",
        "links": "urls",
        "page_list": "urls",
    },
    "search_web": {
        "search_query": "query",
        "search_term": "query",
        "q": "query",
        "search": "query",
        "keywords": "query",
        "num_results": "max_results",
        "n_results": "max_results",
        "count": "max_results",
        "n": "max_results",
        "limit": "max_results",
        "k": "max_results",
    },
    "summarize_page": {
        "text": "content",
        "page_content": "content",
        "body": "content",
        "topic": "focus",
        "query": "focus",
    },
    "write_report": {
        "body": "content",
        "text": "content",
        "report_content": "content",
        "report_title": "title",
        "source_list": "sources",
        "references": "sources",
        "citations": "sources",
        "source_urls": "sources",
        "source_url": "sources",
        "urls": "sources",
        "sources_array": "sources",
    },
}


def _normalize_arg_names(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Переименовать алиасы аргументов в канонические имена."""
    aliases = _ARG_ALIASES.get(tool_name, {})
    if not aliases:
        return kwargs
    return {aliases.get(k, k): v for k, v in kwargs.items()}


def _coerce_scalar(value: Any, prop_type: str | None) -> Any:
    """Привести скалярное значение к типу из JSON Schema (integer/number/boolean)."""
    if prop_type == "integer" and not isinstance(value, int):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if prop_type == "number" and not isinstance(value, (int, float)):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if prop_type == "boolean" and not isinstance(value, bool):
        return str(value).lower() in ("true", "1", "yes")
    return value


def _coerce_array(value: Any) -> list[Any]:
    """Привести значение к списку: JSON-строка → list, скаляр → [value]."""
    import json as _json  # noqa: PLC0415

    if isinstance(value, str):
        # LLM serialised array as JSON string: "[\"a\",\"b\"]"
        try:
            parsed = _json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except _json.JSONDecodeError:
            pass
        # Treat bare string as single-element list
        return [value]

    if not isinstance(value, list):
        return [value]
    return value


def _coerce_args(
    kwargs: dict[str, Any],
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """Привести типы аргументов в соответствие с объявленными в JSON Schema.

    Open-source LLM часто отдают строковые значения для полей integer/array
    или оборачивают массивы в JSON-строки. Здесь они нормализуются перед диспатчем.
    """
    properties: dict[str, Any] = input_schema.get("properties", {})
    result = dict(kwargs)

    for key, value in list(result.items()):
        prop_type = properties.get(key, {}).get("type")
        if prop_type == "array":
            result[key] = _coerce_array(value)
        else:
            result[key] = _coerce_scalar(value, prop_type)

    return result


class ToolRegistry:
    """Центральный реестр: поиск схем + асинхронный диспатч.

    Пример использования:
        registry = ToolRegistry()
        schemas = registry.get_schemas()        # передать LLM
        result = await registry.dispatch("search_web", query="RAG 2024")
    """

    def __init__(self) -> None:
        """Создать реестр: индекс схем по именам + таблица диспатча."""
        self._schemas: dict[str, dict[str, Any]] = {s["name"]: s for s in TOOL_SCHEMAS}
        self._dispatch = TOOL_DISPATCH

    def get_schemas(self) -> list[dict[str, Any]]:
        """Вернуть все схемы инструментов в формате API Anthropic."""
        return list(self._schemas.values())

    def list_tools(self) -> list[str]:
        """Вернуть имена зарегистрированных инструментов."""
        return list(self._dispatch.keys())

    async def dispatch(self, tool_name: str, **kwargs: Any) -> Any:
        """Диспатч вызова инструмента по имени.

        Поднимает ToolError, если имя инструмента не зарегистрировано.
        Пробрасывает ToolError из реализаций инструментов.
        Оборачивает неожиданные исключения в ToolError.
        """
        if tool_name not in self._dispatch:
            raise ToolError(f"Unknown tool: {tool_name}", tool_name=tool_name)

        # 1. Rename aliased argument names (e.g. url_list → urls)
        kwargs = _normalize_arg_names(tool_name, kwargs)
        # 2. Coerce types (e.g. "10" → 10 for integers, "[...]" → list)
        schema = self._schemas.get(tool_name, {})
        kwargs = _coerce_args(kwargs, schema.get("input_schema", {}))

        try:
            return await self._dispatch[tool_name](**kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(str(e), tool_name=tool_name) from e


# ── Register tools ────────────────────────────────────────────────────────────
# Import here (bottom) to avoid circular imports.
# Add new tool functions to TOOL_DISPATCH after implementing them.


def _register_tools() -> None:
    """Заполнить TOOL_DISPATCH всеми функциями инструментов."""
    try:
        from tools.extract_keywords import extract_keywords  # noqa: PLC0415
        from tools.fetch import fetch_pages  # noqa: PLC0415
        from tools.report import write_report  # noqa: PLC0415
        from tools.search import search_web  # noqa: PLC0415
        from tools.summarize import summarize_page  # noqa: PLC0415

        TOOL_DISPATCH.update(
            {
                "search_web": search_web,
                "fetch_pages": fetch_pages,
                "summarize_page": summarize_page,
                "write_report": write_report,
                "extract_keywords": extract_keywords,
            }
        )
    except ImportError as e:
        log.warning("tool_import_failed", error=str(e))


_register_tools()
