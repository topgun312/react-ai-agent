"""LLMClient — мультипровайдерная абстракция над LLM.

Поддерживает:
  - Anthropic (Claude)       через SDK anthropic
  - OpenAI (ChatGPT)         через SDK openai
  - OpenRouter               через SDK openai + custom base_url
  - DeepSeek                 через SDK openai + custom base_url
  - Qwen (DashScope)         через SDK openai + custom base_url
  - MiniMax                  через SDK openai + custom base_url
  - Ollama (локальные модели) через SDK openai + custom base_url
  - GateLLM (gatellm.ru)    через SDK openai + custom base_url
  - Custom endpoint          через SDK openai + CUSTOM_API_BASE_URL

Все клиенты реализуют LLMClientProtocol.
Используйте create_llm_client() для получения нужного клиента из settings.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Protocol, runtime_checkable

import structlog

from config.settings import settings

log = structlog.get_logger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

ToolSchema = dict[str, Any]
LLMResponse = dict[str, Any]  # normalised internal format (Anthropic-style)


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Интерфейс, который должны реализовывать все клиенты LLM."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        system: str = "",
    ) -> LLMResponse:
        """Отправить сообщения и вернуть полный ответ (без стриминга)."""
        ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        system: str = "",
    ) -> AsyncIterator[str]:
        """Отправить сообщения и выдавать текстовые фрагменты по мере поступления."""
        ...


# ── Token budget ──────────────────────────────────────────────────────────────


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Грубая оценка числа токенов: ~4 символа на токен."""
    return len(str(messages)) // 4


def _trim_history(
    messages: list[dict[str, Any]],
    max_tokens: int = 80_000,
) -> list[dict[str, Any]]:
    """Удалить старые сообщения, чтобы не превышать токенный бюджет.

    Всегда сохраняет первое сообщение пользователя (исходный запрос) и
    все сообщения из последних N ходов разговора.
    Не удаляйте и не обходите эту функцию — она предотвращает молчаливую
    обрезку при ошибках API.
    """
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    trimmed = [messages[0]] + messages[1:]
    while len(trimmed) > 2 and _estimate_tokens(trimmed) > max_tokens:
        trimmed.pop(1)

    log.warning("history_trimmed", kept=len(trimmed), original=len(messages))
    return trimmed


# ── Anthropic client ──────────────────────────────────────────────────────────


class AnthropicClient:
    """Продукционный LLM-клиент с использованием SDK Anthropic.

    Обрабатывает нестриминговые completion с JSON-схемами инструментов,
    экспоненциальную повторную попытку при ошибках rate-limit
    и ограничение по токенному бюджету.
    """

    def __init__(
        self,
        model: str | None = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ) -> None:
        """Инициализировать клиент Anthropic.

        Аргументы:
            model:       ID модели. По умолчанию — settings.DEFAULT_MODEL.
            max_retries: Максимум повторных попыток при rate-limit.
            max_tokens:  Лимит токенов в одном ответе.
        """
        self.model = model or settings.DEFAULT_MODEL
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Вернуть (и лениво создать) AsyncAnthropic-клиент SDK."""
        if self._client is None:
            import anthropic  # noqa: PLC0415

            self._client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        system: str = "",
    ) -> LLMResponse:
        """Отправить сообщения и вернуть полный ответ во внутреннем формате.

        Аргументы:
            messages: История сообщений в формате Anthropic.
            tools:    JSON-схемы инструментов (input_schema Anthropic).
            system:   Системный промпт (пустой — без системного промпта).

        Возвращает:
            Словарь с ключами id/content/stop_reason/usage (единый формат
            для всех провайдеров).
        """
        client = self._get_client()
        trimmed = _trim_history(messages)

        async def _call() -> Any:
            """Один нестриминговый запрос к messages.create (для retry)."""
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": trimmed,
            }
            if tools:
                kwargs["tools"] = tools
            if system:
                kwargs["system"] = system
            return await client.messages.create(**kwargs)

        response = await self._with_retry(_call)

        content = []
        for block in response.content:
            if block.type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block.type == "text":
                content.append({"type": "text", "text": block.text})

        return {
            "id": response.id,
            "type": "message",
            "role": response.role,
            "content": content,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        system: str = "",
    ) -> AsyncIterator[str]:
        """Отправить сообщения и выдавать текстовые фрагменты по мере поступления.

        Аргументы:
            messages: История сообщений в формате Anthropic.
            tools:    JSON-схемы инструментов.
            system:   Системный промпт.
        """
        client = self._get_client()
        trimmed = _trim_history(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": trimmed,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system

        async with client.messages.stream(**kwargs) as stream_ctx:
            async for text in stream_ctx.text_stream:
                yield text

    async def _with_retry(self, coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Выполнить корутину с экспоненциальной паузой при rate-limit.

        Аргументы:
            coro_fn: Асинхронная callable для вызова.
            *args:   Позиционные аргументы coro_fn.
            **kwargs: Именованные аргументы coro_fn.

        Возвращает:
            Результат coro_fn.

        Raises:
            Последнее исключение, если все попытки исчерпаны.
        """
        for attempt in range(self.max_retries):
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.max_retries - 1:
                    wait = 2**attempt
                    log.warning("rate_limit_retry", attempt=attempt + 1, wait=wait)
                    await asyncio.sleep(wait)
                else:
                    raise


# ── OpenAI-compatible client ──────────────────────────────────────────────────
#
# Works for: OpenAI, OpenRouter, DeepSeek, Qwen, MiniMax, Ollama
# All of these implement the OpenAI /v1/chat/completions API.

# Base URLs for each provider
_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "minimax": "https://api.minimaxi.chat/v1",
    "ollama": "",  # resolved from OLLAMA_BASE_URL at runtime
    "gatellm": "https://gatellm.ru/v1",
    "custom": "",  # resolved from CUSTOM_API_BASE_URL at runtime
}


# API key per provider (resolved from settings at runtime)
def _get_api_key(provider: str) -> str:
    """Вернуть API-ключ провайдера из settings (для Ollama — заглушка)."""
    key_map = {
        "openai": settings.OPENAI_API_KEY,
        "openrouter": settings.OPENROUTER_API_KEY,
        "deepseek": settings.DEEPSEEK_API_KEY,
        "qwen": settings.QWEN_API_KEY,
        "minimax": settings.MINIMAX_API_KEY,
        "ollama": "ollama",  # Ollama doesn't require a real key
        "gatellm": settings.GATELLM_API_KEY,
        "custom": settings.CUSTOM_API_KEY,
    }
    return key_map.get(provider, "")


# Обязательный API-ключ для каждого провайдера (пусто = ключ не требуется)
_REQUIRED_KEY_FIELDS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "QWEN_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "gatellm": "GATELLM_API_KEY",
    "ollama": "",  # локальный сервер, ключ не нужен
    "custom": "",  # ключ опционален
}


def _validate_provider_config(provider: str) -> None:
    """Проверить, что для выбранного провайдера задан нужный API-ключ."""
    field = _REQUIRED_KEY_FIELDS.get(provider)
    if field and not getattr(settings, field, ""):
        raise ValueError(
            f"LLM_PROVIDER={provider!r}, but {field} is not set. "
            f"Add {field}=... to your .env (see .env.example)."
        )


def _anthropic_tools_to_openai(tools: list[ToolSchema]) -> list[dict[str, Any]]:
    """Преобразовать схему инструментов Anthropic в формат function calling OpenAI."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def _anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: str,
) -> list[dict[str, Any]]:
    """Преобразовать формат сообщений Anthropic в формат сообщений OpenAI.

    Ключевые отличия:
    - Системный промпт в OpenAI — это отдельное сообщение роли 'system'
    - Блоки tool_use в сообщениях ассистента → поле tool_calls
    - Блоки tool_result в сообщениях пользователя → отдельные сообщения роли 'tool'
    """
    result: list[dict[str, Any]] = []

    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        # content is a list of blocks
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
        tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
        text_blocks = [b for b in content if b.get("type") == "text"]

        if tool_use_blocks:
            # Assistant message with tool calls
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    },
                }
                for b in tool_use_blocks
            ]
            text = text_blocks[0]["text"] if text_blocks else None
            result.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                }
            )

        elif tool_result_blocks:
            # User message with tool results → one 'tool' message per result
            for b in tool_result_blocks:
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": b["tool_use_id"],
                        "content": b.get("content", ""),
                    }
                )

        elif text_blocks:
            text = "\n".join(b["text"] for b in text_blocks)
            result.append({"role": role, "content": text})

    return result


def _openai_response_to_anthropic(response: Any) -> LLMResponse:
    """Привести ответ OpenAI chat completion к внутреннему формату Anthropic."""
    choice = response.choices[0]
    message = choice.message
    finish_reason = choice.finish_reason  # "stop" | "tool_calls" | "length"

    content: list[dict[str, Any]] = []

    if message.content:
        content.append({"type": "text", "text": message.content})

    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                input_data = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                input_data = {"raw": tc.function.arguments}
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": input_data,
                }
            )

    # Map OpenAI finish_reason → Anthropic stop_reason
    stop_reason_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }
    stop_reason = stop_reason_map.get(finish_reason or "stop", "end_turn")

    usage = response.usage
    return {
        "id": response.id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.model,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        },
    }


class OpenAICompatibleClient:
    """LLM-клиент для OpenAI-совместимых API.

    Работает с: OpenAI, OpenRouter, DeepSeek, Qwen, MiniMax, Ollama.
    Приводит запросы/ответы к тому же внутреннему формату, что и AnthropicClient.
    """

    def __init__(
        self,
        provider: str,
        model: str | None = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ) -> None:
        """Инициализировать OpenAI-совместимый клиент.

        Аргументы:
            provider:    Ключ провайдера (openai, ollama, deepseek, ...).
            model:       ID модели. По умолчанию — settings.DEFAULT_MODEL.
            max_retries: Максимум повторных попыток при rate-limit.
            max_tokens:  Лимит токенов в одном ответе.
        """
        self.provider = provider
        self.model = model or settings.DEFAULT_MODEL
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        """Вернуть (и лениво создать) AsyncOpenAI-клиент с base_url провайдера."""
        if self._client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            base_url = _PROVIDER_BASE_URLS.get(self.provider, "")
            if self.provider == "ollama":
                base_url = settings.OLLAMA_BASE_URL.rstrip("/") + "/v1"
            elif self.provider == "custom":
                base_url = settings.CUSTOM_API_BASE_URL.rstrip("/")

            api_key = _get_api_key(self.provider)

            kwargs: dict[str, Any] = {"api_key": api_key or "none"}
            if base_url:
                kwargs["base_url"] = base_url

            # OpenRouter requires a Referer header
            if self.provider == "openrouter":
                kwargs["default_headers"] = {
                    "HTTP-Referer": "https://github.com/research-agent",
                    "X-Title": "Research Agent",
                }

            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        system: str = "",
    ) -> LLMResponse:
        """Отправить сообщения и вернуть полный ответ во внутреннем формате.

        Конвертирует формат Anthropic → OpenAI, выполняет запрос и приводит
        ответ обратно к внутреннему (Anthropic-подобному) формату.

        Аргументы:
            messages: История сообщений в формате Anthropic.
            tools:    JSON-схемы инструментов (input_schema Anthropic).
            system:   Системный промпт.

        Возвращает:
            Словарь с ключами id/content/stop_reason/usage.
        """
        client = self._get_client()
        trimmed = _trim_history(messages)
        oai_messages = _anthropic_messages_to_openai(trimmed, system)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": oai_messages,
        }
        if tools:
            kwargs["tools"] = _anthropic_tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        for attempt in range(self.max_retries):
            try:
                response = await client.chat.completions.create(**kwargs)
                return _openai_response_to_anthropic(response)
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < self.max_retries - 1:
                    wait = 2**attempt
                    log.warning(
                        "rate_limit_retry",
                        provider=self.provider,
                        attempt=attempt + 1,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        raise RuntimeError("unreachable")  # pragma: no cover

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema],
        system: str = "",
    ) -> AsyncIterator[str]:
        """Отправить сообщения и выдавать текстовые фрагменты по мере поступления.

        Аргументы:
            messages: История сообщений в формате Anthropic.
            tools:    JSON-схемы инструментов.
            system:   Системный промпт.
        """
        client = self._get_client()
        trimmed = _trim_history(messages)
        oai_messages = _anthropic_messages_to_openai(trimmed, system)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": oai_messages,
            "stream": True,
        }

        async with await client.chat.completions.create(**kwargs) as stream_ctx:
            async for chunk in stream_ctx:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


# ── Factory ───────────────────────────────────────────────────────────────────


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClientProtocol:
    """Создать нужный LLM-клиент на основе настроек провайдера.

    Аргументы:
        provider: Переопределение settings.LLM_PROVIDER (например, из флага --provider CLI).
        model:    Переопределение settings.DEFAULT_MODEL (например, из флага --model CLI).

    Возвращает:
        AnthropicClient или OpenAICompatibleClient в зависимости от провайдера.
    """
    p = provider or settings.LLM_PROVIDER
    m = model or settings.DEFAULT_MODEL

    _validate_provider_config(p)

    if p == "anthropic":
        return AnthropicClient(model=m)

    if p in _PROVIDER_BASE_URLS:
        return OpenAICompatibleClient(provider=p, model=m)

    raise ValueError(
        f"Unknown provider: {p!r}. "
        f"Choose from: anthropic, {', '.join(_PROVIDER_BASE_URLS)}"
    )
