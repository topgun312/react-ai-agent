"""Orchestrator — ReAct-цикл: Рассуждение → Действие → Наблюдение → повтор.

Центральный компонент. Любые изменения здесь должны сохранять пять
инвариантов цикла, описанных в CLAUDE.md в разделе "ReAct Loop — Invariants".
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from agent.llm_client import LLMClientProtocol, create_llm_client
from agent.state import AgentState, Message, Source
from config.settings import settings
from tools.registry import ToolError, ToolRegistry

log = structlog.get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a research assistant. Your job is to research a topic and produce a report.

RULES — follow them strictly:
1. You MUST use tools. Do NOT answer from memory alone.
2. You MUST end every session by calling the write_report tool.
3. Never stop with a plain text response — always call a tool.
4. The user's query is about AI/ML unless stated otherwise.
   "RAG" means Retrieval-Augmented Generation (an AI technique), not anything else.

WORKFLOW:
Step 1 — Call search_web with a specific query to find sources.
Step 2 — Call fetch_pages on the most promising URLs (pick 2-4).
Step 3 — If any page is very long, call summarize_page on it.
Step 4 — Call write_report with a structured Markdown report and all sources.

IMPORTANT:
- Always call write_report at the end. This is mandatory.
- Include source URLs in the report.
- Be concise but thorough.
"""


# ── Orchestrator ──────────────────────────────────────────────────────────────


class Orchestrator:
    """Запускает ReAct-цикл для исследовательского запроса.

    Инварианты (должны сохраняться после любых изменений):
    1. Лимит шагов соблюдается — цикл выходит после max_steps даже без write_report
    2. Каждый вызов инструмента логируется на уровне INFO
    3. ToolError не роняет цикл — добавляется как результат tool
    4. Состояние append-only — сообщения не удаляются
    5. write_report всегда завершает цикл — выход сразу
    """

    def __init__(
        self,
        llm: LLMClientProtocol | None = None,
        registry: ToolRegistry | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Инициализировать оркестратор.

        Аргументы:
            llm:       Клиент LLM. По умолчанию — create_llm_client() из settings.
            registry:  Реестр инструментов. По умолчанию — новый ToolRegistry.
            max_steps: Лимит шагов цикла. По умолчанию — settings.MAX_STEPS.
        """
        self.llm = llm or create_llm_client()
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps or settings.MAX_STEPS

    async def run(self, query: str) -> AgentState:
        """Выполняет полный ReAct-цикл для исследовательского запроса.

        Возвращает итоговое состояние AgentState с заполненными полями report и sources.
        """
        state = AgentState(query=query)
        state.append_message(self._build_initial_message(query))

        tools = self.registry.get_schemas()

        while state.step < self.max_steps:
            state.increment_step()
            log.info("react_step", step=state.step, max_steps=self.max_steps)

            response = await self.llm.complete(
                messages=state.to_api_messages(),
                tools=tools,
                system=SYSTEM_PROMPT,
            )

            # Append assistant turn to history (invariant 4: append-only)
            state.append_message(Message(role="assistant", content=response["content"]))

            stop_reason = response.get("stop_reason", "")

            # end_turn: LLM stopped without calling a tool.
            if stop_reason == "end_turn":
                if self._handle_end_turn(state, response, query) == "continue":
                    continue  # give the model one more chance
                break

            if stop_reason != "tool_use":
                log.warning(
                    "unexpected_stop_reason", stop_reason=stop_reason, step=state.step
                )
                break

            tool_results, done = await self._process_tool_blocks(state, response)

            # Append all tool results as a single user message (Anthropic API requirement)
            if tool_results:
                state.append_message(Message(role="user", content=tool_results))

            if done:
                log.info("react_done", reason="write_report", step=state.step)
                break
        else:
            log.warning("react_max_steps_reached", step=state.step)

        return state

    def _handle_end_turn(
        self,
        state: AgentState,
        response: dict[str, Any],
        query: str,
    ) -> str:
        """Обработать stop_reason == "end_turn".

        Возвращает:
            "continue" — дать LLM ещё один шанс (напоминание про write_report),
            "stop"     — завершить цикл.
        """
        text_parts = [b["text"] for b in response["content"] if b.get("type") == "text"]

        if text_parts and not state.report:
            # LLM wrote the report as plain text instead of calling write_report.
            raw_text = "\n\n".join(text_parts)
            state.report = f"# Research: {query}\n\n{raw_text}"
            log.warning(
                "fallback_report_from_text",
                step=state.step,
                chars=len(state.report),
            )
            return "stop"

        if not state.report and state.step < self.max_steps:
            # LLM returned nothing — inject a hard reminder and loop again.
            log.warning("end_turn_no_content_retry", step=state.step)
            state.append_message(
                Message(
                    role="user",
                    content=(
                        "You stopped without calling write_report. "
                        "You MUST call write_report now to finish the research. "
                        "Use all the information you have gathered so far."
                    ),
                )
            )
            return "continue"

        log.info("llm_end_turn", step=state.step)
        return "stop"

    async def _process_tool_blocks(
        self,
        state: AgentState,
        response: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Выполнить все tool_use блоки ответа LLM.

        Возвращает:
            tool_results — блоки tool_result для сообщения пользователю,
            done         — True, если write_report успешно записал отчёт
                           (инвариант 5: завершает цикл сразу).
        """
        tool_results: list[dict[str, Any]] = []
        done = False

        for block in response["content"]:
            if block.get("type") != "tool_use":
                continue

            tool_name: str = block["name"]
            tool_input: dict[str, Any] = block["input"]
            tool_use_id: str = block["id"]

            result_str = await self._dispatch_tool(
                state, tool_name, tool_input, tool_use_id
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str,
                }
            )

            # Invariant 5: write_report exits immediately — but only if
            # it actually succeeded. A failed write_report (e.g. bad args)
            # must be retried, not terminate the loop.
            if tool_name == "write_report":
                done = state.report is not None
                if not done:
                    log.warning("write_report_failed_retry", step=state.step)
                break

        return tool_results, done

    async def _dispatch_tool(
        self,
        state: AgentState,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
    ) -> str:
        """Выполняет один вызов инструмента и возвращает результат строкой.

        Ловит ToolError и возвращает его как строку ошибки (никогда не пробрасывает).
        Логирует каждый вызов на уровне INFO.
        Обновляет state.sources, если инструмент вернул данные об источниках.
        """
        # Invariant 2: every dispatch logged
        log.info(
            "tool_dispatched",
            tool=tool_name,
            step=state.step,
            input_preview=str(tool_input)[:120],
        )

        try:
            result = await self.registry.dispatch(tool_name, **tool_input)

            if tool_name == "write_report":
                # result is a ReportResult — extract content and sources
                state.report = result.content
                for src in result.sources:
                    state.add_source(
                        Source(
                            url=src.get("url", ""),
                            title=src.get("title", ""),
                        )
                    )
                return f"Report written: {result.title} ({result.word_count} words)"

            if tool_name == "search_web" and isinstance(result, list):
                for r in result:
                    if r.get("url"):
                        state.add_source(
                            Source(
                                url=r["url"],
                                title=r.get("title", ""),
                                snippet=r.get("snippet", ""),
                            )
                        )

            # Convert result to a readable string for the LLM
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)

        except ToolError as e:
            # Invariant 3: ToolError never crashes loop
            log.warning("tool_error", tool=tool_name, error=str(e), step=state.step)
            return f"Error executing {tool_name}: {e}"

    def _build_initial_message(self, query: str) -> Message:
        """Собрать стартовое сообщение пользователя с темой исследования."""
        return Message(role="user", content=f"Research topic: {query}")
