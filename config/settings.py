"""Settings — Pydantic Settings с загрузкой из .env.

Валидирует все обязательные переменные окружения при запуске.
Импортируйте `settings` из этого модуля — не создавайте Settings напрямую.

Пример использования:
    from config.settings import settings
    print(settings.DEFAULT_MODEL)
    print(settings.LLM_PROVIDER)
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

# All supported LLM providers
LLMProvider = Literal[
    "anthropic",
    "openai",
    "openrouter",
    "deepseek",
    "qwen",
    "minimax",
    "ollama",
    "gatellm",  # https://gatellm.ru — OpenAI-compatible gateway
    "custom",  # any OpenAI-compatible endpoint via CUSTOM_API_BASE_URL
]


class Settings(BaseSettings):
    """Вся конфигурация Research Agent.

    Значения загружаются из переменных окружения и файла .env.
    Установите LLM_PROVIDER, чтобы выбрать бэкенд LLM.
    """

    # ── Provider selector ─────────────────────────────────────────────────────

    LLM_PROVIDER: LLMProvider = Field(
        default="ollama",
        description=(
            "Which LLM provider to use. "
            "Options: anthropic, openai, openrouter, deepseek, qwen, minimax, "
            "ollama, gatellm, custom"
        ),
    )
    # ── API keys (set only the ones you need) ────────────────────────────────
    ANTHROPIC_API_KEY: str = Field(
        default="",
        description="Anthropic Claude API key. Required when LLM_PROVIDER=anthropic.",
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key. Required when LLM_PROVIDER=openai.",
    )
    OPENROUTER_API_KEY: str = Field(
        default="",
        description="OpenRouter API key. Required when LLM_PROVIDER=openrouter.",
    )
    DEEPSEEK_API_KEY: str = Field(
        default="",
        description="DeepSeek API key. Required when LLM_PROVIDER=deepseek.",
    )
    QWEN_API_KEY: str = Field(
        default="",
        description="Alibaba DashScope API key. Required when LLM_PROVIDER=qwen.",
    )
    MINIMAX_API_KEY: str = Field(
        default="",
        description="MiniMax API key. Required when LLM_PROVIDER=minimax.",
    )
    GATELLM_API_KEY: str = Field(
        default="",
        description="GateLLM API key (gatellm.ru). Required when LLM_PROVIDER=gatellm.",
    )

    # ── Custom / generic OpenAI-compatible endpoint ───────────────────────────
    CUSTOM_API_BASE_URL: str = Field(
        default="",
        description=(
            "Base URL for any OpenAI-compatible API. "
            "Used when LLM_PROVIDER=custom. Example: https://my-llm.example.com/v1"
        ),
    )
    CUSTOM_API_KEY: str = Field(
        default="",
        description="API key for the custom endpoint. Used when LLM_PROVIDER=custom.",
    )

    # ── Local / Ollama ────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL. Used when LLM_PROVIDER=ollama.",
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    DEFAULT_MODEL: str = Field(
        default="qwen2.5:7b",
        description=(
            "Model ID to use. Must match the chosen provider. "
            "Examples: claude-sonnet-4-6, gpt-4o, deepseek-chat, qwen-plus, "
            "qwen/qwen-2.5-72b-instruct (gatellm/openrouter), qwen2.5:7b (ollama)."
        ),
    )

    # ── Agent loop ────────────────────────────────────────────────────────────
    MAX_STEPS: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Hard limit on ReAct loop iterations.",
    )
    # ── HTTP ──────────────────────────────────────────────────────────────────
    REQUEST_TIMEOUT: int = Field(
        default=30,
        ge=5,
        le=120,
        description="HTTP request timeout in seconds.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    REPORTS_DIR: str = Field(
        default="research/",
        description="Directory where finished reports are saved.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR.",
    )


# Singleton instance — import this everywhere
settings = Settings()
