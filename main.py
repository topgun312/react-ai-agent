"""main.py — CLI-точка входа для Research Agent.

Примеры использования:
    python main.py "Ваша тема исследования"
    python main.py "тема" --provider deepseek --model deepseek-chat --save
    python main.py "тема" --provider ollama --model qwen2.5:7b --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_PROVIDERS = [
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
    "qwen",
    "minimax",
    "ollama",
    "gatellm",
    "custom",
]


def _configure_logging(level: str) -> None:
    """Настройка structlog с консольным рендерером логов."""
    import logging  # noqa: PLC0415

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Собрать CLI-парсер с флагами провайдера, модели и лимита шагов."""
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Autonomous research agent using ReAct loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
            providers:
            anthropic   Claude (claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5)
            openai      ChatGPT (gpt-4o, gpt-4o-mini, o1, o3-mini)
            openrouter  Any model via OpenRouter (e.g. meta-llama/llama-3.3-70b-instruct)
            deepseek    DeepSeek (deepseek-chat, deepseek-reasoner)
            qwen        Alibaba Qwen (qwen-plus, qwen-turbo, qwen-max)
            minimax     MiniMax (MiniMax-Text-01)
            ollama      Local models via Ollama (qwen2.5:7b, mistral, gemma3, etc.)
            gatellm     GateLLM gateway (gatellm.ru) — set GATELLM_API_KEY
            custom      Any OpenAI-compatible endpoint — set CUSTOM_API_BASE_URL + CUSTOM_API_KEY


            examples:
              python main.py "RAG best practices" --provider anthropic
              python main.py "RAG best practices" --provider deepseek --model deepseek-chat
              python main.py "RAG best practices" --provider ollama --model qwen2.5:7b
              python main.py "RAG best practices" --provider openrouter --model meta-llama/llama-3.3-70b-instruct
              python main.py "RAG best practices" --provider gatellm --model qwen/qwen-2.5-72b-instruct
        """,
    )
    parser.add_argument("query", help="Research topic or question")
    parser.add_argument(
        "--provider",
        choices=_PROVIDERS,
        default=None,
        help="LLM provider to use (overrides LLM_PROVIDER in .env)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID for the chosen provider (overrides DEFAULT_MODEL in .env)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of ReAct loop steps (overrides MAX_STEPS in .env)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save report to REPORTS_DIR after completion",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    """Главная асинхронная точка входа. Возвращает код завершения."""
    from agent.llm_client import create_llm_client  # noqa: PLC0415
    from agent.orchestrator import Orchestrator  # noqa: PLC0415
    from config.settings import settings  # noqa: PLC0415
    from ui.display import print_banner, print_report, print_summary  # noqa: PLC0415

    # Apply CLI overrides to settings singleton
    if args.provider:
        settings.LLM_PROVIDER = args.provider
    if args.model:
        settings.DEFAULT_MODEL = args.model
    if args.max_steps:
        settings.MAX_STEPS = args.max_steps

    print_banner(provider=settings.LLM_PROVIDER, model=settings.DEFAULT_MODEL)

    llm = create_llm_client(
        provider=settings.LLM_PROVIDER,
        model=settings.DEFAULT_MODEL,
    )
    orchestrator = Orchestrator(llm=llm)

    try:
        state = await orchestrator.run(args.query)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        return 130

    print_report(state=state)
    print_summary(state=state)

    if args.save and state.report:
        _save_report(state, settings.REPORTS_DIR)

    return 0


def _save_report(state: Any, reports_dir: str) -> None:
    """Сохранить отчёт в Markdown-файл в каталоге REPORTS_DIR."""
    from ui.display import console  # noqa: PLC0415

    dir_path = Path(reports_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^\w\s-]", "", state.query).strip()
    safe_name = re.sub(r"\s+", "_", safe_name)[:60] or "report"
    file_path = dir_path / f"{safe_name}.md"

    file_path.write_text(state.report, encoding="utf-8")
    log.info("report_saved", path=str(file_path))
    console.print(f"\n[green]Report saved →[/green] {file_path}")


def main() -> None:
    """Точка входа: разобрать аргументы, запустить агент, вывести отчёт."""
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging("DEBUG" if args.verbose else "INFO")

    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
