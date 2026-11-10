"""
Factory responsável pela criação dos provedores de IA.
"""

import os

from ai_analyst.llm.anthropic import AnthropicClient
from ai_analyst.llm.base import LLMProvider
from ai_analyst.llm.gemini import GeminiClient
from ai_analyst.llm.openai import OpenAIClient
from ai_analyst.llm.resilience import RetryLLMProvider

AVAILABLE_PROVIDERS = (
    "gemini",
    "openai",
    "anthropic",
)


def create_llm_provider(
    provider_name: str | None = None,
) -> LLMProvider:
    """
    Cria o provedor de IA selecionado e adiciona resiliência.

    O provedor pode ser definido pelo argumento provider_name
    ou pela variável de ambiente LLM_PROVIDER.

    Gemini permanece como provedor padrão.
    """

    configured_provider = (
        provider_name
        if provider_name is not None
        else os.getenv("LLM_PROVIDER", "gemini")
    )

    provider = configured_provider.strip().lower()

    if not provider:
        provider = "gemini"

    if provider == "gemini":
        llm_provider = GeminiClient()

    elif provider == "openai":
        llm_provider = OpenAIClient()

    elif provider == "anthropic":
        llm_provider = AnthropicClient()

    else:
        available = ", ".join(AVAILABLE_PROVIDERS)

        raise ValueError(
            f"Provedor de IA não suportado: '{provider}'. "
            f"Provedores disponíveis: {available}."
        )

    return RetryLLMProvider(llm_provider)
