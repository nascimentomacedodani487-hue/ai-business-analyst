"""
Camada de resiliência para provedores de modelos de linguagem.

Implementa novas tentativas com backoff exponencial
para falhas temporárias.
"""

import logging
from time import sleep
from typing import Callable

from ai_analyst.llm.base import (
    LLMProvider,
    LLMTimeoutError,
    LLMTransientProviderError,
)
from ai_analyst.llm.metrics import llm_metrics

logger = logging.getLogger(__name__)


class RetryLLMProvider:
    """
    Envolve um provedor de IA e adiciona retry inteligente.

    Apenas falhas temporárias são repetidas:
    - LLMTimeoutError
    - LLMTransientProviderError

    Erros de cota, autenticação, resposta inválida
    e outras falhas permanentes são propagados imediatamente.
    """

    def __init__(
        self,
        provider: LLMProvider,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        sleep_fn: Callable[[float], None] = sleep,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts deve ser maior ou igual a 1.")

        if initial_delay < 0:
            raise ValueError("initial_delay não pode ser negativo.")

        self.provider = provider
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.sleep_fn = sleep_fn

    @property
    def provider_name(self) -> str:
        """Retorna o identificador do provedor original."""

        return getattr(
            self.provider,
            "provider_name",
            "gemini",
        )

    @property
    def model(self) -> str:
        """Retorna o modelo utilizado pelo provedor original."""

        return getattr(
            self.provider,
            "model",
            "unknown",
        )

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        max_output_tokens: int = 512,
    ) -> str:
        """Executa uma geração textual com retry."""

        return self._execute_with_retry(
            lambda: self.provider.generate_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_output_tokens=max_output_tokens,
            )
        )

    def generate_json_response(
        self,
        user_prompt: str,
        system_prompt: str,
        response_schema: dict,
        max_output_tokens: int = 2048,
    ) -> str:
        """
        Executa geração JSON estruturada com retry.

        Quando o provedor oferece generate_json_response,
        utiliza esse recurso.

        Para provedores sem suporte específico, utiliza
        generate_response como alternativa compatível.
        """

        structured_method = getattr(
            self.provider,
            "generate_json_response",
            None,
        )

        if callable(structured_method):
            return self._execute_with_retry(
                lambda: structured_method(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    response_schema=response_schema,
                    max_output_tokens=max_output_tokens,
                )
            )

        return self.generate_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
        )

    def _execute_with_retry(
        self,
        generate: Callable[[], str],
    ) -> str:
        """Executa uma operação com retry e backoff exponencial."""

        for attempt in range(1, self.max_attempts + 1):
            try:
                return generate()

            except (
                LLMTimeoutError,
                LLMTransientProviderError,
            ) as exc:
                if attempt >= self.max_attempts:
                    logger.error(
                        "Tentativas de chamada ao provedor esgotadas",
                        extra={
                            "event": "llm_retry_exhausted",
                            "attempt": attempt,
                            "max_attempts": self.max_attempts,
                            "error_type": type(exc).__name__,
                        },
                    )
                    raise

                delay = self.initial_delay * (2 ** (attempt - 1))

                llm_metrics.record_retry()

                logger.warning(
                    "Falha temporária no provedor. Nova tentativa programada.",
                    extra={
                        "event": "llm_retry_scheduled",
                        "attempt": attempt,
                        "next_attempt": attempt + 1,
                        "max_attempts": self.max_attempts,
                        "delay_seconds": delay,
                        "error_type": type(exc).__name__,
                    },
                )

                self.sleep_fn(delay)

        raise RuntimeError("Fluxo inesperado na camada de resiliência.")
