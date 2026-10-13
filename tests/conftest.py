"""conftest.py — общие фикстуры для всех тестов.

Предоставляет:
- MockLLMClient: заранее запланированные последовательности tool_use для интеграционных тестов
- env_override: временная подстановка переменных окружения
- sample_html: реалистичная HTML-страница для тестов fetch
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Generator
from unittest.mock import patch

import pytest

# ── MockLLMClient ─────────────────────────────────────────────────────────────


class MockLLMClient:
    """LLM-клиент, возвращающий заранее запланированные ответы.

    Настройте `responses` как список словарей в формате ответа Anthropic.
    Каждый вызов `complete()` извлекает следующий ответ из списка.

    Пример использования:
        client = MockLLMClient([
            _tool_use("search_web", {"query": "test"}, "tu_001"),
            _tool_use("write_report", {"title": "T", "content": "C"}, "tu_002"),
        ])
        orchestrator = Orchestrator(llm=client)
        state = await orchestrator.run("test")
        assert state.report is not None
    """

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        """Инициализировать мок с подготовленной очередью ответов.

        Аргументы:
            responses: Список ответов в формате Anthropic; при исчерпании
                       очереди возвращается моковый write_report.
        """
        self._responses = list(responses or [])
        self._call_count = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
        system: str = "",
    ) -> dict[str, Any]:
        """Вернуть следующий запланированный ответ (или write_report по умолчанию)."""
        self._call_count += 1
        if not self._responses:
            # Default: return write_report to terminate loop
            return _tool_use_response(
                "write_report",
                {
                    "title": "Test Report",
                    "content": "# Test\n\nDefault mock report.",
                    "sources": [],
                },
                f"tu_default_{self._call_count}",
            )
        return self._responses.pop(0)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any],
        system: str = "",
    ) -> AsyncIterator[str]:
        """Отдать один фиктивный фрагмент стриминга."""
        yield "Mock streaming response"


def _tool_use_response(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_use_id: str = "tu_001",
) -> dict[str, Any]:
    """Собрать мок-ответ API Anthropic с блоком tool_use."""
    return {
        "id": f"msg_{tool_use_id}",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": tool_input,
            }
        ],
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Мок LLM по умолчанию: search → write_report."""
    return MockLLMClient(
        [
            _tool_use_response("search_web", {"query": "test query"}, "tu_001"),
            _tool_use_response(
                "write_report",
                {
                    "title": "Test Report",
                    "content": "# Test Report\n\nFindings here.\n",
                    "sources": [{"url": "https://example.com", "title": "Example"}],
                },
                "tu_002",
            ),
        ]
    )


@pytest.fixture
def sample_html() -> str:
    """Реалистичная HTML-страница для тестов fetch_pages."""
    return """<!DOCTYPE html>
<html>
<head><title>Test Page — RAG Best Practices</title></head>
<body>
  <nav>Navigation should be stripped</nav>
  <main>
    <h1>RAG Best Practices 2024</h1>
    <p>Retrieval-Augmented Generation combines dense retrieval with generation.</p>
    <p>Key techniques: hybrid search, reranking, and chunk size optimization.</p>
  </main>
  <script>alert('should be stripped')</script>
  <style>.hidden { display: none; }</style>
</body>
</html>"""


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подставить фиктивные API-ключи, чтобы валидация settings прошла в тестах."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    # Also patch the already-instantiated settings singleton
    from config.settings import settings  # noqa: PLC0415

    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant-test-key")


@pytest.fixture(autouse=True)
def mock_ddg() -> Generator[None, None, None]:
    """Предотвращать реальные обращения к DuckDuckGo во всех тестах.

    Отдельные тесты, проверяющие search_web, могут переопределить это своим патчем.
    """
    fake_results = [
        {"href": "https://example.com", "title": "Example", "body": "search result"},
    ]
    with patch("ddgs.DDGS.text", return_value=fake_results):
        yield
