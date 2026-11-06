
"""
Integração com a API oficial da Anthropic (Claude).
"""

import logging
import os
from time import perf_counter

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Anthropic,
    RateLimitError,
)

from dotenv import load_dotenv

from ai_analyst.llm.base import (
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTransientProviderError,
)
from ai_analyst.llm.metrics import llm_metrics


load_dotenv()

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Gerencia a comunicação com os modelos Claude."""

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
    ):
        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY não encontrada. "
                "Verifique o arquivo .env."
            )

        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        max_output_tokens: int = 512,
    ) -> str:
        """Gera uma resposta utilizando a API da Anthropic."""

        start_time = perf_counter()
        prompt_length = len(user_prompt)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            response_text = "\n".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
                and getattr(block, "text", None)
            ).strip()

            if not response_text:
                raise LLMResponseError(
                    "A Anthropic retornou uma resposta vazia."
                )

            duration_ms = round(
                (perf_counter() - start_time) * 1000,
                2,
            )

            usage = getattr(response, "usage", None)

            prompt_tokens = getattr(
                usage,
                "input_tokens",
                None,
            )

            output_tokens = getattr(
                usage,
                "output_tokens",
                None,
            )

            total_tokens = (
                prompt_tokens + output_tokens
                if prompt_tokens is not None
                and output_tokens is not None
                else None
            )

            llm_metrics.record_success(
                provider=self.provider_name,
                model=self.model,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

            logger.info(
                "Chamada à Anthropic concluída",
                extra={
                    "event": "anthropic_request_completed",
                    "provider": self.provider_name,
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "prompt_length": prompt_length,
                    "response_length": len(response_text),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            )

            return response_text

        except Exception as exc:
            duration_ms = round(
                (perf_counter() - start_time) * 1000,
                2,
            )

            llm_metrics.record_failure(
                provider=self.provider_name,
                model=self.model,
                duration_ms=duration_ms,
            )

            logger.exception(
                "Falha na chamada à Anthropic",
                extra={
                    "event": "anthropic_request_failed",
                    "provider": self.provider_name,
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "prompt_length": prompt_length,
                },
            )

            if isinstance(exc, LLMError):
                raise

            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            if isinstance(exc, RateLimitError) or status_code == 429:
                raise LLMRateLimitError(
                    "O limite de requisições da Anthropic "
                    "foi atingido."
                ) from exc

            if (
                isinstance(exc, (APITimeoutError, TimeoutError))
                or status_code == 408
            ):
                raise LLMTimeoutError(
                    "A comunicação com a Anthropic excedeu "
                    "o tempo limite."
                ) from exc

            if (
                isinstance(exc, APIConnectionError)
                or status_code in (500, 502, 503, 504)
            ):
                raise LLMTransientProviderError(
                    "O serviço Anthropic está temporariamente "
                    "indisponível."
                ) from exc

            if isinstance(exc, APIStatusError):
                raise LLMProviderError(
                    f"A API Anthropic retornou um erro HTTP "
                    f"{status_code}."
                ) from exc

            raise LLMProviderError(
                "Falha na comunicação com o provedor Anthropic."
            ) from exc
