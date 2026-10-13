"""display.py — Rich-интерфейс для Research Agent.

Обрабатывает весь вывод в терминале:
- Анимированный спиннер во время размышлений LLM
- Отображение прогресса шагов с именем инструмента и аргументами
- Итоговый отчёт, отображённый как Markdown
- Форматирование списка источников

Единственный модуль, которому разрешён прямой вывод Rich-консоли (не structlog).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agent.state import AgentState

console = Console()


def print_banner(provider: str = "", model: str = "") -> None:
    """Вывести баннер запуска Research Agent."""
    info = "[bold blue]Research Agent[/bold blue]  ·  v0.1  ·  DataTalks.ru"
    if provider and model:
        info += f"\n[dim]Provider: {provider}  ·  Model: {model}[/dim]"
    console.print(Panel(info, border_style="dim blue"))


def print_step(step: int, tool_name: str, tool_input: dict[str, Any]) -> None:
    """Вывести заголовок шага при вызове инструмента."""
    # Show up to 2 key args, truncated for readability
    preview_items = list(tool_input.items())[:2]
    args_str = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in preview_items)
    console.print(
        f"\n[bold cyan]Step {step}[/bold cyan] → [yellow]{tool_name}[/yellow]({args_str})"
    )


@contextmanager
def thinking_spinner(message: str = "Thinking...") -> Generator[None, None, None]:
    """Менеджер контекста: показывает спиннер, пока LLM думает."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task(message, total=None)
        yield


def print_report(state: AgentState) -> None:
    """Отобразить итоговый отчёт как Markdown в терминале."""
    if not state.report:
        console.print("[red]No report generated.[/red]")
        return

    console.print("\n")
    console.rule("[bold blue]Research Report[/bold blue]")
    console.print(Markdown(state.report))
    print_sources(state)


def print_sources(state: AgentState) -> None:
    """Вывести форматированную таблицу всех источников, найденных за исследование."""
    if not state.sources:
        return

    table = Table(
        title="Sources",
        show_header=True,
        header_style="bold blue",
        show_lines=False,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="green", no_wrap=False)
    table.add_column("URL", style="cyan", no_wrap=False)

    for i, source in enumerate(state.sources, 1):
        table.add_row(str(i), source.title or "—", source.url)

    console.print("\n")
    console.print(table)


def print_summary(state: AgentState) -> None:
    """Вывести итоги сессии: использовано шагов, вызовов инструментов, найдено источников."""
    console.print(
        f"\n[dim]Steps: {state.step}  ·  "
        f"Sources: {len(state.sources)}  ·  "
        f"Messages: {len(state.messages)}[/dim]"
    )
