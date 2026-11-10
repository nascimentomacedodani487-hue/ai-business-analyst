"""
Contratos e exceções comuns para provedores de modelos de linguagem.
"""

from typing import Protocol, runtime_checkable


class LLMError(Exception):
    """Erro-base da camada de provedores de IA."""


class LLMProviderError(LLMError):
    """Falha na comunicação ou execução de um provedor de IA."""


class LLMTransientProviderError(LLMProviderError):
    """Falha temporária do provedor que pode permitir nova tentativa."""


class LLMRateLimitError(LLMProviderError):
    """Limite de requisições ou cota do provedor atingido."""


class LLMTimeoutError(LLMProviderError):
    """Tempo limite de comunicação com o provedor excedido."""


class LLMResponseError(LLMProviderError):
    """Resposta vazia, inválida ou inesperada do provedor."""


@runtime_checkable
class LLMProvider(Protocol):
    """Contrato comum para todos os provedores de IA."""

    @property
    def provider_name(self) -> str:
        """Identificador do provedor."""
        ...

    @property
    def model(self) -> str:
        """Identificador do modelo utilizado."""
        ...

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        max_output_tokens: int = 512,
    ) -> str:
        """Gera uma resposta a partir de instruções e de um prompt."""
        ...
