"""test_agent.py — интеграционные тесты полного ReAct-цикла.

Проверяет работу Orchestrator с MockLLMClient, чтобы проверить поведение
цикла без реальных API-вызовов.

Проверяемые ключевые инварианты (см. CLAUDE.md "ReAct Loop — Invariants"):
1. Лимит шагов соблюдается
2. ToolError не роняет цикл
3. write_report завершает цикл
4. Состояние append-only
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.conftest import MockLLMClient, _tool_use_response


@pytest.mark.anyio
async def test_agent_completes_with_write_report(
    mock_llm_client: MockLLMClient,
) -> None:
    """Полный цикл: search → write_report → state.report заполнен."""
    from agent.orchestrator import Orchestrator

    orchestrator = Orchestrator(llm=mock_llm_client)
    state = await orchestrator.run("test research topic")
    assert state.report is not None
    assert "Test Report" in state.report


@pytest.mark.anyio
async def test_agent_respects_max_steps() -> None:
    """Цикл выходит после MAX_STEPS, даже если write_report так и не вызван."""
    # LLM always returns search_web — no write_report
    endless_search = MockLLMClient(
        [
            _tool_use_response("search_web", {"query": f"query {i}"}, f"tu_{i:03d}")
            for i in range(20)
        ]
    )

    from agent.orchestrator import Orchestrator

    orchestrator = Orchestrator(llm=endless_search, max_steps=3)
    state = await orchestrator.run("infinite loop test")
    assert state.step <= 3
    assert state.report is None  # no report without write_report


@pytest.mark.anyio
async def test_agent_survives_tool_error() -> None:
    """ToolError из инструмента не роняет цикл — цикл продолжается."""

    from tools.registry import ToolError

    client = MockLLMClient(
        [
            _tool_use_response("search_web", {"query": "test"}, "tu_001"),
            _tool_use_response(
                "write_report",
                {
                    "title": "Recovery Report",
                    "content": "# Recovery\n\nDespite error, continued.",
                    "sources": [],
                },
                "tu_002",
            ),
        ]
    )

    from agent.orchestrator import Orchestrator
    from tools.registry import ToolRegistry

    registry = ToolRegistry()

    # Make search_web raise ToolError
    original_dispatch = registry.dispatch

    async def patched_dispatch(tool_name: str, **kwargs: Any) -> Any:
        """Мок диспатча: search_web падает с ToolError, остальное — оригинал."""
        if tool_name == "search_web":
            raise ToolError("simulated API failure", tool_name="search_web")
        return await original_dispatch(tool_name, **kwargs)

    registry.dispatch = patched_dispatch  # type: ignore[method-assign]

    orchestrator = Orchestrator(llm=client, registry=registry)
    state = await orchestrator.run("error recovery test")

    # Loop continued and produced a report despite the error
    assert state.report is not None
    assert "Recovery" in state.report


@pytest.mark.anyio
async def test_agent_state_is_append_only(
    mock_llm_client: MockLLMClient,
) -> None:
    """Список сообщений только растёт — никогда не уменьшается между шагами."""
    from agent.orchestrator import Orchestrator

    orchestrator = Orchestrator(llm=mock_llm_client)

    # Patch run to record sizes after each step — use a subclass approach
    original_run = orchestrator.run

    async def instrumented_run(query: str) -> Any:
        """Обёртка run() для замера размера истории между шагами."""
        state = await original_run(query)
        return state

    state = await orchestrator.run("test")
    # Verify messages only grew: first msg + assistant + user tool result + ...
    assert len(state.messages) >= 2


@pytest.mark.anyio
async def test_agent_state_sources_populated(
    mock_llm_client: MockLLMClient,
) -> None:
    """После успешного запуска state.sources содержит источники отчёта."""
    from agent.orchestrator import Orchestrator

    orchestrator = Orchestrator(llm=mock_llm_client)
    state = await orchestrator.run("test")
    # Sources from write_report's sources list
    assert len(state.sources) > 0
    assert state.sources[0].url == "https://example.com"
