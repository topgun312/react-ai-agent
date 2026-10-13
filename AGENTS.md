# Research Agent — AGENTS.md

> AI coding assistant instructions for Claude Code, Codex, and similar tools.
> Read this file before making any changes to the codebase.

---

## Project Overview

Research Agent is a CLI tool that autonomously researches any topic by searching
the web, fetching and summarizing pages in parallel, and synthesizing a structured
Markdown report with citations.

**Stack:** Python 3.11+, httpx (async HTTP), Anthropic/OpenAI SDK, Pydantic, Rich.
**Pattern:** ReAct loop — the LLM decides which tools to call; the orchestrator
executes them and returns results until `write_report` is invoked.

---

## Architecture

```
CLI (main.py)
  └─► Orchestrator (agent/orchestrator.py)   ← ReAct loop, step counter, stop condition
        ├─► AgentState (agent/state.py)       ← message history, scratchpad, sources list
        ├─► LLMClient (agent/llm_client.py)   ← Anthropic/OpenAI SDK, streaming, retry
        └─► ToolRegistry (tools/registry.py)  ← tool registration, JSON Schema, dispatch
              ├─► search_web   (tools/search.py)    → Tavily API
              ├─► fetch_pages  (tools/fetch.py)     → httpx async + BeautifulSoup
              ├─► summarize_page (tools/summarize.py) → LLM compression
              └─► write_report (tools/report.py)    → LLM + Markdown formatter
```

Data flow:
1. User query → `main.py` → `Orchestrator.run(query)`
2. Orchestrator sends messages + tool schemas to LLM
3. LLM returns tool_call → Orchestrator dispatches via ToolRegistry
4. Tool result appended to AgentState → next LLM call
5. LLM calls `write_report` → loop exits → report printed via Rich

---

## Key Files

| File | Role | Notes |
|------|------|-------|
| `agent/orchestrator.py` | ReAct loop entry point | Central file — changes here affect all tool execution |
| `agent/state.py` | `AgentState` dataclass | Holds full conversation + scratchpad + source list |
| `agent/llm_client.py` | LLM abstraction | Supports Anthropic and OpenAI; handles streaming + retries |
| `tools/registry.py` | Tool registration + dispatch | **All new tools must be registered here with JSON Schema** |
| `tools/search.py` | `search_web` tool | Calls Tavily REST API |
| `tools/fetch.py` | `fetch_pages` tool | Async concurrent HTTP fetch + HTML parsing |
| `tools/summarize.py` | `summarize_page` tool | LLM-based page compression |
| `tools/report.py` | `write_report` tool | Terminal condition — produces final Markdown report |
| `config/settings.py` | Pydantic Settings | Loads `.env`, validates API keys on startup |
| `ui/display.py` | Rich terminal output | Progress bars, spinners, Markdown rendering |

---

## Setup & Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Then fill in: ANTHROPIC_API_KEY, TAVILY_API_KEY

# Run the agent
python -m research_agent "Best practices for building RAG systems in 2024"

# Run with options
python -m research_agent "topic" --model claude-sonnet-4-6 --max-steps 15

# Run tests
pytest tests/ -x -v

# Run linter + type checker
ruff check . && mypy agent/ tools/
```

---

## Code Conventions

### Async everywhere
All I/O operations (HTTP requests, LLM calls) **must use async/await**.
`fetch_pages` uses `asyncio.gather()` for concurrent fetching — do not convert to sync.

### Adding a new tool
1. Create `tools/your_tool.py` with an `async def your_tool(...)` function
2. Add full JSON Schema to `tools/registry.py` in `TOOL_SCHEMAS`
3. Register the function in `TOOL_DISPATCH` dict in `tools/registry.py`
4. Add unit test in `tests/test_tools.py` with mocked HTTP/LLM

### Logging
Use `structlog` for all logging — **never use `print()` in production code**.
Exception: `ui/display.py` is allowed to use Rich console directly.

```python
import structlog
log = structlog.get_logger()
log.info("tool_called", tool_name="search_web", query=query)
```

### Type hints
All public functions and methods require type hints. Private helpers (_prefixed) are optional.

### Error handling
Tools raise `ToolError(message, tool_name)` on recoverable failures.
The orchestrator catches `ToolError`, appends the error as a tool result, and continues the loop.
Do not raise bare `Exception` from tool code.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | Claude API key |
| `OPENAI_API_KEY` | Yes (if using OpenAI) | OpenAI API key |
| `TAVILY_API_KEY` | Yes | Web search API key |
| `DEFAULT_MODEL` | No | Default: `claude-sonnet-4-6` |
| `MAX_STEPS` | No | Default: `10` — hard stop for ReAct loop |
| `REQUEST_TIMEOUT` | No | Default: `30` seconds per HTTP request |
| `LOG_LEVEL` | No | Default: `INFO`. Set `DEBUG` for full tool traces |

**Security:** Never commit `.env` or any file containing API keys.
The `.gitignore` already excludes `.env` — do not remove this rule.

---

## Testing

### Unit tests (`tests/test_tools.py`)
Test individual tools in isolation. Mock all external HTTP with `respx`:

```python
import respx, httpx

@respx.mock
async def test_fetch_pages():
    respx.get("https://example.com").mock(
        return_value=httpx.Response(200, text="<h1>Hello</h1>")
    )
    result = await fetch_pages(["https://example.com"])
    assert "Hello" in result[0].content
```

### Integration tests (`tests/test_agent.py`)
Test the full ReAct loop with a mocked LLM. Use `MockLLMClient` from `tests/conftest.py`
which returns pre-scripted tool_call sequences:

```python
async def test_agent_completes(mock_llm_client):
    orchestrator = Orchestrator(llm=mock_llm_client)
    result = await orchestrator.run("test query")
    assert result.report is not None
    assert len(result.sources) > 0
```

### What NOT to mock
- **Never mock `ToolRegistry`** — test real tool dispatch to catch schema mismatches
- **Never mock `AgentState`** — it's a simple dataclass, no need
- **Do mock** external APIs: Tavily, Anthropic/OpenAI, any HTTP endpoints

---

## Anti-Patterns

### 1. Sync HTTP in async tools
```python
# WRONG — blocks the event loop
def fetch_pages(urls):
    return [requests.get(url).text for url in urls]

# CORRECT — concurrent async
async def fetch_pages(urls: list[str]) -> list[PageContent]:
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[_fetch_one(client, url) for url in urls])
```

### 2. Tool registered without JSON Schema
Every tool dispatched by the LLM **must have a complete JSON Schema** in `TOOL_SCHEMAS`.
Missing or wrong schema causes the LLM to hallucinate tool arguments.

### 3. Mutable default in AgentState
```python
# WRONG — shared mutable state across instances
@dataclass
class AgentState:
    messages: list = []  # shared reference!

# CORRECT
@dataclass
class AgentState:
    messages: list = field(default_factory=list)
```

### 4. Exceeding context window silently
`llm_client.py` enforces a token budget. Do not remove or bypass `_trim_history()` —
it exists to prevent silent truncation errors from the API.

### 5. Hardcoding model names
Never hardcode `"claude-sonnet-4-6"` in tool or orchestrator code.
Always read from `settings.DEFAULT_MODEL`.

### 6. print() in agent loop
The orchestrator runs in a streaming context. Unexpected `print()` calls corrupt
Rich's live display. Use `structlog` or pass messages through `display.py`.

---

## ReAct Loop — Key Invariants

These invariants must hold after any change to `orchestrator.py`:

1. **Step limit is always enforced** — loop exits after `settings.MAX_STEPS` even if no `write_report`
2. **Every tool call is logged** — `log.info("tool_dispatch", tool=name, step=step)`
3. **Tool errors do not crash the loop** — `ToolError` is caught and appended as tool result
4. **State is append-only** — never mutate or delete existing messages in `AgentState.messages`
5. **write_report always terminates** — receiving `write_report` in tool_use must exit the loop

---

## Common Workflows

### Adding search capabilities to an existing tool
1. Inject `ToolRegistry` into the component that needs it (constructor injection)
2. Call `await registry.dispatch(tool_name, **kwargs)` — do not call tools directly
3. Update JSON Schema in `registry.py` if tool signature changes

### Changing the LLM provider
Swap `agent/llm_client.py` implementation. The `LLMClientProtocol` defines the interface:
```python
class LLMClientProtocol(Protocol):
    async def complete(self, messages: list[Message], tools: list[ToolSchema]) -> LLMResponse: ...
    async def stream(self, messages: list[Message], tools: list[ToolSchema]) -> AsyncIterator[str]: ...
```
Both `AnthropicClient` and `OpenAIClient` implement this protocol.

### Debugging a stuck ReAct loop
1. Set `LOG_LEVEL=DEBUG` in `.env`
2. Run with `--max-steps 3` to reproduce quickly
3. Check structlog output for `tool_dispatch` events — missing events = LLM not calling tools
4. Inspect `AgentState.scratchpad` for LLM reasoning traces

---

## Project Status

- [x] Core ReAct loop
- [x] Tool registry with JSON Schema validation
- [x] Parallel page fetching (asyncio.gather)
- [x] Streaming LLM responses
- [x] Rich terminal UI
- [ ] Persistent report storage (planned)
- [ ] Web UI (planned)
- [ ] Multi-agent delegation (planned for Module 03)