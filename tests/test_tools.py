"""test_tools.py — юнит-тесты отдельных инструментов.

Каждый инструмент проверяется изолированно с замоканными внешними зависимостями.
НЕ мокайте ToolRegistry — тестируйте реальный диспатч, чтобы ловить
расхождения схем с реализацией.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from tools.registry import ToolError, ToolRegistry

# ── search_web ────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_search_web_returns_results() -> None:
    """search_web возвращает список результатов с url, title, snippet."""
    mock_raw = [
        {"href": "https://example.com", "title": "Example", "body": "snippet text"},
    ]
    with patch("tools.search._ddg_search", return_value=mock_raw[:1]):
        # _ddg_search уже возвращает нормализованный формат — нет, он возвращает raw
        pass

    # Патчим на уровне DDGS.text (вызывается внутри _ddg_search в потоке)
    with patch("ddgs.DDGS.text", return_value=mock_raw):
        from tools.search import search_web

        results = await search_web("test query", max_results=1)

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com"
    assert results[0]["title"] == "Example"
    assert results[0]["snippet"] == "snippet text"


@pytest.mark.anyio
async def test_search_web_empty_results() -> None:
    """search_web возвращает пустой список при отсутствии результатов."""
    with patch("ddgs.DDGS.text", return_value=[]):
        from tools.search import search_web

        results = await search_web("very obscure query no results")
    assert results == []


@pytest.mark.anyio
async def test_search_web_raises_tool_error_on_ratelimit() -> None:
    """search_web оборачивает RatelimitException в ToolError."""
    from ddgs.exceptions import RatelimitException

    with patch("ddgs.DDGS.text", side_effect=RatelimitException("rate limit")):
        from tools.search import search_web

        with pytest.raises(ToolError):
            await search_web("test query")


@pytest.mark.anyio
async def test_search_web_raises_tool_error_on_generic_exception() -> None:
    """search_web оборачивает любое неожиданное исключение в ToolError."""
    with patch("ddgs.DDGS.text", side_effect=RuntimeError("network error")):
        from tools.search import search_web

        with pytest.raises(ToolError):
            await search_web("test query")


# ── fetch_pages ───────────────────────────────────────────────────────────────


@pytest.mark.anyio
@respx.mock
async def test_fetch_pages_extracts_text(sample_html: str) -> None:
    """fetch_pages извлекает текст, вырезая скрипты и блоки навигации."""
    respx.get("https://example.com/").mock(
        return_value=httpx.Response(200, text=sample_html)
    )

    from tools.fetch import fetch_pages

    results = await fetch_pages(["https://example.com/"])
    assert len(results) == 1
    assert "RAG Best Practices" in results[0]["content"]
    assert "alert(" not in results[0]["content"]  # script stripped


@pytest.mark.anyio
@respx.mock
async def test_fetch_pages_handles_failed_url() -> None:
    """fetch_pages добавляет элемент с ошибкой для упавших URL и не поднимает исключение."""
    respx.get("https://bad-url.example/").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    from tools.fetch import fetch_pages

    results = await fetch_pages(["https://bad-url.example/"])
    assert results[0].get("error") is not None


@pytest.mark.anyio
@respx.mock
async def test_fetch_pages_concurrent(sample_html: str) -> None:
    """fetch_pages загружает несколько URL параллельно."""
    for i in range(3):
        respx.get(f"https://example{i}.com/").mock(
            return_value=httpx.Response(200, text=sample_html)
        )

    from tools.fetch import fetch_pages

    results = await fetch_pages([f"https://example{i}.com/" for i in range(3)])
    assert len(results) == 3
    assert all("content" in r for r in results)


# ── write_report ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_write_report_returns_formatted_markdown() -> None:
    """write_report собирает контент и источники в ReportResult."""
    from tools.report import write_report

    result = await write_report(
        title="Test Report",
        content="# Test\n\nSome findings.",
        sources=[{"url": "https://example.com", "title": "Example"}],
    )

    assert result.title == "Test Report"
    assert "## References" in result.content
    assert "https://example.com" in result.content
    assert result.word_count > 0
    assert len(result.sources) == 1


@pytest.mark.anyio
async def test_write_report_no_sources() -> None:
    """write_report корректно работает при пустом списке источников."""
    from tools.report import write_report

    result = await write_report(title="T", content="# T\n\nContent.", sources=[])
    assert "## References" not in result.content
    assert result.word_count > 0


# ── ToolRegistry dispatch ─────────────────────────────────────────────────────


def test_registry_lists_all_tools() -> None:
    """ToolRegistry перечисляет все ожидаемые инструменты."""
    registry = ToolRegistry()
    tools = registry.list_tools()
    assert "search_web" in tools
    assert "fetch_pages" in tools
    assert "summarize_page" in tools
    assert "write_report" in tools


def test_registry_schemas_have_descriptions() -> None:
    """У каждой схемы инструмента есть description для всех обязательных свойств."""
    registry = ToolRegistry()
    for schema in registry.get_schemas():
        assert (
            "description" in schema
        ), f"{schema['name']} missing top-level description"
        for prop_name, prop in schema["input_schema"]["properties"].items():
            assert (
                "description" in prop
            ), f"{schema['name']}.{prop_name} missing property description"


@pytest.mark.anyio
async def test_registry_raises_tool_error_on_unknown_tool() -> None:
    """Реестр поднимает ToolError для неизвестных имён инструментов."""
    registry = ToolRegistry()
    with pytest.raises(ToolError, match="Unknown tool"):
        await registry.dispatch("unknown_tool", query="test")


@pytest.mark.anyio
async def test_extract_keywords_returns_list() -> None:
    """extract_keywords возвращает список строк не длиннее max_keywords."""
    from tools.extract_keywords import extract_keywords

    result = await extract_keywords(
        "Retrieval-Augmented Generation combines retrieval with generation models.",
        max_keywords=5,
    )
    assert isinstance(result, list)
    assert len(result) <= 5
    assert all(isinstance(k, str) for k in result)


@pytest.mark.anyio
async def test_extract_keywords_raises_on_empty() -> None:
    """Пустой текст вызывает ToolError."""
    from tools.extract_keywords import extract_keywords

    with pytest.raises(ToolError):
        await extract_keywords("")


@pytest.mark.anyio
async def test_extract_keywords_via_registry() -> None:
    """extract_keywords доступен через ToolRegistry.dispatch."""
    registry = ToolRegistry()
    result = await registry.dispatch(
        "extract_keywords", text="Machine learning models."
    )
    assert isinstance(result, list)
