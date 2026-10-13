"""AgentState — история сообщений, блокнот и список источников для ReAct-цикла.

Единый источник всего, что знает агент за сессию.
Состояние append-only: никогда не удаляйте и не мутируйте существующие сообщения.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Message types ─────────────────────────────────────────────────────────────

Role = Literal["user", "assistant"]


@dataclass
class ToolUse:
    """Представляет один вызов инструмента со стороны LLM."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """Результат, возвращённый из инструмента."""

    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """Одно сообщение в истории разговора."""

    role: Role
    content: str | list[Any]  # str for text, list for tool_use/tool_result blocks


@dataclass
class Source:
    """Веб-источник, цитируемый в исследовании."""

    url: str
    title: str
    snippet: str = ""


# ── AgentState ────────────────────────────────────────────────────────────────


@dataclass
class AgentState:
    """Полное состояние одной исследовательской сессии.

    Хранит историю переписки в формате сообщений Anthropic,
    блокнот для промежуточных трасс рассуждений LLM
    и дедуплицированный список источников, найденных за время исследования.
    """

    query: str
    messages: list[Message] = field(default_factory=list)
    scratchpad: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    step: int = 0
    report: str | None = None

    def append_message(self, message: Message) -> None:
        """Добавить сообщение в историю разговора (только добавление)."""
        self.messages.append(message)

    def add_source(self, source: Source) -> None:
        """Добавить источник, дедуплицируя по URL."""
        if not any(s.url == source.url for s in self.sources):
            self.sources.append(source)

    def to_api_messages(self) -> list[dict[str, Any]]:
        """Преобразовать сообщения в формат, ожидаемый API Anthropic."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def increment_step(self) -> None:
        """Увеличить счётчик шагов на единицу."""
        self.step += 1

    def __repr__(self) -> str:
        """Компактное представление состояния для отладки."""
        return (
            f"AgentState(query={self.query!r}, "
            f"steps={self.step}, "
            f"messages={len(self.messages)}, "
            f"sources={len(self.sources)})"
        )
