# Research Agent

Автономный CLI-агент, который исследует произвольную тему: ищет источники
в интернете, параллельно скачивает страницы, извлекает контент и на выходе
генерирует структурированный Markdown-отчёт со ссылками на источники.
Без фреймворков — только Python, httpx и LLM SDK.

Ядро — **ReAct-цикл** (Reason → Act → Observe): LLM сама решает, что делать
дальше, оркестратор исполняет выбранные инструменты и возвращает результат,
и так до тех пор, пока LLM не вызовет `write_report`.

---

## Общая схема работы

```
┌─────────────────────────────────────────────────────────────────────┐
│                     python main.py "Тема запроса"                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  config/settings.py   │  LLM_PROVIDER, DEFAULT_MODEL,
                    │  (Pydantic Settings)  │  MAX_STEPS, API-ключи (.env)
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  agent/llm_client.py  │  создаёт клиента для провайдера
                    │  create_llm_client()  │  (anthropic / openai-compatible)
                    └───────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────────────────┐
        │              agent/orchestrator.py — ReAct-цикл           │
        │                                                           │
        │  ┌────────┐   сообщения + сх. инструментов                │
        │  │  LLM   │◄─────────────────────────────────┐            │
        │  └───┬────┘                                  │            │
        │      │ (tool_use: search_web /              │            │
        │      │  fetch_pages / write_report ...)     │            │
        │      ▼                                      │            │
        │  ┌─────────────────────┐                    │            │
        │  │  tools/registry.py  │  dispatch()        │            │
        │  └─────────┬───────────┘  ▪ алиасы аргументов │            │
        │            │             ▪ приведение типов  │            │
        │            ▼                                   │            │
        │  ┌───────────────────────┐   результат        │            │
        │  │  tools/*.py — инструмент│───────────────────┘            │
        │  └───────────────────────┘   (сохраняется в AgentState)    │
        │                                                           │
        │  write_report → цикл завершается, отчёт сохранён в state  │
        └───────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  ui/display.py (Rich) │  баннер, отчёт в Markdown,
                    │                       │  таблица источников
                    └───────────────────────┘
```

### Поток данных по шагам

1. `main.py` парсит аргументы CLI, применяет переопределения к `settings`,
   создаёт LLM-клиент через `create_llm_client()` и запускает
   `Orchestrator.run(query)`.
2. Оркестратор создаёт `AgentState`, отправляет LLM системный промпт +
   сообщение с темой + JSON-Schema всех инструментов.
3. LLM возвращает либо `tool_use` (решила вызвать инструмент), либо `end_turn`
   (закончила диалог).
4. Если `tool_use` — оркестратор диспатчит инструмент через `ToolRegistry`,
   результат добавляется в историю сообщений, цикл повторяется.
5. Если LLM вызвала `write_report` и отчёт записан — цикл останавливается.
6. `main.py` выводит отчёт через Rich и при `--save` сохраняет в файл.

---

## Инструменты агента

Пять инструментов покрывают полный цикл исследования. Каждый описывается
JSON Schema (`tools/registry.py`), агент сам решает, когда и какой использовать.

| Инструмент | Что делает | Источник данных |
|------------|------------|-----------------|
| `search_web(query)` | Поиск релевантных URL по запросу. Возвращает title, url, snippet для топ-N результатов. | DuckDuckGo (`ddgs`, без API-ключа) |
| `fetch_pages(urls)` | Загружает несколько URL параллельно через `asyncio.gather()`. Извлекает plain text из HTML (BeautifulSoup). Обрезает до 3000 символов. | httpx async |
| `summarize_page(content, focus)` | Сжатие длинного текста LLM-ом до 200–400 слов вокруг заданного фокуса (защита от переполнения контекста). | LLM (провайдер из `.env`) |
| `extract_keywords(text)` | Локальное извлечение частотных ключевых слов из текста (`Counter`, без LLM). | — |
| `write_report(title, content, sources)` | **Терминальный** инструмент: принимает готовый Markdown от LLM, добавляет секцию References и возвращает `ReportResult`. Сам LLM не вызывает — контент пишет модель в tool_use. | — |

### Функциональность

- **Автономный ReAct Loop** — агент сам решает, сколько шагов нужно: ищет,
  читает, ищет снова, синтезирует.
- **Параллельная загрузка** — несколько URL загружаются одновременно
  через `asyncio.gather()`.
- **Цитаты в отчёте** — каждый факт привязан к источнику с заголовком и URL.
- **Обработка ошибок** — упавший инструмент не убивает агента: `ToolError`
  передаётся LLM как `tool_result`, модель решает, как продолжить.
- **Выбор провайдера** — Anthropic, OpenAI, OpenRouter, DeepSeek, Qwen,
  MiniMax, Ollama (локально), GateLLM, custom endpoint — всё через `.env`
  и флаги CLI.

---

## Структура проекта

```
research_agent/
├── main.py              # точка входа, CLI (argparse)
├── agent/
│   ├── orchestrator.py  # ReAct loop, stop condition
│   ├── state.py         # AgentState: messages, scratchpad, sources
│   └── llm_client.py    # Anthropic/OpenAI совместимые клиенты, retry, streaming
├── tools/
│   ├── registry.py      # регистрация + JSON Schema + dispatch
│   ├── search.py        # search_web → DuckDuckGo (ddgs)
│   ├── fetch.py         # fetch_pages → httpx async + BeautifulSoup
│   ├── summarize.py     # summarize_page → LLM
│   ├── extract_keywords.py  # extract_keywords → Counter (без LLM)
│   └── report.py        # write_report → Markdown formatter
├── ui/
│   └── display.py       # Rich: баннер, spinner, Markdown render
├── config/
│   └── settings.py      # Pydantic Settings, .env загрузка
├── research/            # папка для сохранённых отчётов
├── tests/
│   ├── conftest.py      # MockLLMClient, фикстуры
│   ├── test_tools.py    # unit-тесты инструментов (mock HTTP)
│   └── test_agent.py    # интеграционный тест loop (mock LLM)
├── pyproject.toml
└── .env.example         # LLM_PROVIDER, API-ключи провайдеров
```

| Модуль | Роль |
|--------|------|
| `main.py` | CLI-вход. Парсит аргументы (`--provider`, `--model`, `--max-steps`, `--save`, `--verbose`), запускает агента, печатает и сохраняет отчёт |
| `agent/orchestrator.py` | **Центр всего.** ReAct-цикл: вызывает LLM, диспатчит инструменты, следит за лимитом шагов и завершением через `write_report`. Все инварианты цикла живут здесь |
| `agent/state.py` | `AgentState` — состояние одной сессии: история сообщений, число шагов, список источников, готовый отчёт. Append-only |
| `agent/llm_client.py` | Абстракция над LLM. `AnthropicClient` (SDK Anthropic) и `OpenAICompatibleClient` (OpenAI SDK + свой base_url) для всех остальных провайдеров. Приводит ответы к единому внутреннему формату и обрезает историю по токенному бюджету |
| `tools/registry.py` | Реестр инструментов: хранит JSON-Schema каждого инструмента и таблицу диспатча. Нормализует аргументы (алиасы имён + приведение типов) и заворачивает любые сбои в `ToolError` |
| `tools/search.py` | `search_web` — веб-поиск через DuckDuckGo (библиотека `ddgs`, без API-ключа). Синхронная библиотека запускается в потоке через `asyncio.to_thread()` |
| `tools/fetch.py` | `fetch_pages` — параллельное скачивание страниц (`asyncio.gather` + httpx) и извлечение читаемого текста через BeautifulSoup. Один упавший URL не валит всё |
| `tools/summarize.py` | `summarize_page` — сжатие длинного текста LLM-ом. Использует общий `create_llm_client()`, т.е. работает с любым провайдером из `.env` |
| `tools/extract_keywords.py` | `extract_keywords` — локальное извлечение частотных ключевых слов (Counter, без LLM) |
| `tools/report.py` | `write_report` — **терминальный** инструмент: принимает готовый Markdown от LLM, добавляет секцию References и возвращает `ReportResult`. Сам LLM не вызывает |
| `config/settings.py` | Pydantic Settings: все настройки и API-ключи из `.env`, выбор провайдера (`LLM_PROVIDER`), лимит шагов, таймауты |
| `ui/display.py` | Rich-интерфейс: баннер, спиннер, отчёт как Markdown, таблица источников. Единственный модуль, которому разрешён прямой вывод в консоль |
| `tests/` | Юнит-тесты инструментов (мок внешних HTTP) и интеграционные тесты ReAct-цикла с поддельным LLM |

### Модель данных сессии (`agent/state.py`)

```
AgentState
 ├── query: str                     — исходный запрос пользователя
 ├── messages: list[Message]        — вся переписка (append-only):
 │                                     Message(role: user|assistant,
 │                                             content: str | list[block])
 │                                     блоки: text | tool_use | tool_result
 ├── scratchpad: list[str]          — трассы рассуждений LLM
 ├── sources: list[Source]          — найденные источники (дедупликация по URL):
 │                                     Source(url, title, snippet)
 ├── step: int                      — номер текущего шага цикла
 └── report: str | None             — итоговый отчёт (заполняется write_report)
```

---

## Технологический стек

- Python 3.12+
- anthropic SDK
- openai SDK
- asyncio
- httpx[http2]
- beautifulsoup4
- ddgs (поиск DuckDuckGo)
- pydantic + pydantic-settings
- rich
- structlog
- pytest + respx (тесты), black + isort + flake8 (качество кода)

---

## Ключевые механизмы

### ReAct-цикл (`agent/orchestrator.py`, метод `run`)

Инварианты, которые нельзя нарушать при изменении:

1. **Лимит шагов всегда соблюдается** — цикл выходит после `settings.MAX_STEPS`,
   даже если `write_report` так и не вызван.
2. **Каждый вызов инструмента логируется** (structlog, уровень INFO).
3. **Ошибка инструмента не убивает цикл** — `ToolError` ловится и добавляется
   в историю как `tool_result`, модель может понять ошибку и повторить.
4. **Состояние append-only** — сообщения в `AgentState.messages` не удаляются
   и не мутируются.
5. **`write_report` завершает цикл** — но только если отчёт действительно
   записан (`state.report is not None`). Упавший `write_report` не останавливает
   агента — ошибка возвращается модели, и цикл продолжается.

Особый случай: если LLM вернула `end_turn`, а отчёт ещё не написан — оркестратор
инжектирует напоминание «Ты обязан вызвать write_report» и даёт модели ещё шанс.
Если модель написала отчёт обычным текстом без вызова инструмента — текст
заворачивается в Markdown как fallback.

### Реестр инструментов (`tools/registry.py`)

- `TOOL_SCHEMAS` — JSON-Schema каждого инструмента на языке Anthropic
  (`input_schema`). Отправляется LLM, чтобы она вызывала инструменты
  с правильными аргументами.
- `TOOL_DISPATCH` — словарь `имя_инструмента → async-функция`.
  Оркестратор **никогда** не вызывает инструменты напрямую, только
  через `registry.dispatch()`.
- `_ARG_ALIASES` — маппинг «вольных» имён аргументов, которые выдумывают
  open-source модели, к каноническим. Например, для `write_report`:
  `body`/`text`/`report_content` → `content`, `report_title` → `title`,
  `source_list`/`references`/`citations` → `sources`. Без этого
  llama-подобные модели падают с TypeError.
- `_coerce_args` — приведение типов по схеме: строки вида `"5"` → `5` для
  integer, JSON-строки `"[\"a\"]"` → список для array.
- `dispatch()` ловит все исключения инструмента и заворачивает в `ToolError`.

### LLM-провайдеры (`agent/llm_client.py`)

Провайдер выбирается переменной `LLM_PROVIDER` в `.env` — все запросы
(оркестратор и `summarize_page`) идут через один выбранный провайдер.

Единый внутренний формат ответа — «как у Anthropic». Все OpenAI-совместимые
провайдеры (`OpenAICompatibleClient`) приводятся к этому формату:

- Anthropic `stop_reason` ↔ OpenAI `finish_reason`
  (`tool_use` ↔ `tool_calls`, `end_turn` ↔ `stop`)
- блоки `tool_use` ↔ поле `tool_calls`
- блоки `tool_result` ↔ отдельные сообщения роли `tool`
- JSON-Schema инструментов → формат OpenAI function calling

Поддерживаемые провайдеры: `anthropic`, `openai`, `openrouter`, `deepseek`,
`qwen`, `minimax`, `ollama` (локально), `gatellm`, `custom` (любая
OpenAI-совместимая точка). Base URL и API-ключ берутся из `settings`;
при отсутствии обязательного ключа `create_llm_client()` сразу сообщает
об ошибке. `_trim_history()` следит за токенным бюджетом: при переполнении
удаляются самые старые сообщения, но всегда сохраняется исходный запрос.

---

## Установка и запуск

```bash
# 1. Установка зависимостей
pip install -e .

# 2. Конфигурация (.env в корне проекта)
cp .env.example .env
#   Раскомментировать нужный провайдер и заполнить ключи.
#   Для Ollama ключ не нужен — достаточно запущенного сервера
#   на http://localhost:11434 и модели (например: ollama pull qwen2.5:7b).

# 3. Запуск
python main.py "Вопросы собеседования по Python"

# С выбором провайдера/модели
python main.py "тема" --provider deepseek --model deepseek-chat --save
python main.py "тема" --provider ollama --model qwen2.5:7b --verbose
python main.py "тема" --provider openrouter --model meta-llama/llama-3.3-70b-instruct
```

### Ключевые настройки (`.env`)

| Переменная | Значение по умолчанию | Назначение |
|------------|----------------------|------------|
| `LLM_PROVIDER` | `ollama` | Провайдер LLM |
| `DEFAULT_MODEL` | `qwen2.5:7b` | Модель провайдера |
| `MAX_STEPS` | `10` | Лимит итераций ReAct-цикла |
| `REQUEST_TIMEOUT` | `30` | Таймаут HTTP-запросов (сек) |
| `REPORTS_DIR` | `research/` | Куда сохраняются отчёты при `--save` |
| `LOG_LEVEL` | `INFO` | Уровень логов (`DEBUG` — полные трассы инструментов) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Адрес локального Ollama |

### CLI-флаги

| Флаг | Назначение |
|------|------------|
| `--provider` | Переопределить провайдера LLM |
| `--model` | Переопределить модель |
| `--max-steps` | Лимит шагов ReAct-цикла |
| `--save` | Сохранить отчёт в `REPORTS_DIR` |
| `--verbose` | Включить DEBUG-логирование |

---

## Как добавить новый инструмент

1. Создать `tools/my_tool.py` с `async def my_tool(...)` (с type-хинтами
   и docstring). Ошибки поднимать как `ToolError(message, tool_name="my_tool")`.
2. Добавить JSON-Schema в `TOOL_SCHEMAS` в `tools/registry.py`.
3. Зарегистрировать функцию в `TOOL_DISPATCH` (в `_register_tools()`, внизу файла).
4. При желании добавить алиасы аргументов в `_ARG_ALIASES`.
5. Написать юнит-тест в `tests/test_tools.py` с моком внешних HTTP.

---

## Тесты и качество кода

```bash
pytest tests/ -x -v        # юнит + интеграционные тесты
flake8 agent config tests tools ui main.py
isort agent config tests tools ui main.py
black agent config tests tools ui main.py
```

Правила тестирования:

- инструменты тестируются изолированно, внешний HTTP мокается (`respx`);
- полный ReAct-цикл тестируется с поддельным `MockLLMClient` из
  `tests/conftest.py`, который возвращает заранее запланированные
  последовательности `tool_use`;
- `ToolRegistry` и `AgentState` мокать нельзя — реальная диспетчеризация
  ловит расхождения схемы с реализацией.

---

## Известные особенности

- `tools/search.py` использует **DuckDuckGo** (библиотека `ddgs`,
  бесплатно, без API-ключа).
- Модели через Ollama/локальные часто «вольничают» с именами аргументов —
  для этого и существует слой нормализации в `registry.py`.
- Запас прочности по контексту: страницы обрезаются до 3000 символов
  в `fetch.py`, история сообщений — лимитом 80 000 «токенов» в `llm_client.py`.
