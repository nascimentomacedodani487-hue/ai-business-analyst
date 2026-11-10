"""
Integração com a API oficial da OpenAI.
"""

import logging
import os
from time import perf_counter

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

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


class OpenAIClient:
    """Gerencia a comunicação com os modelos da OpenAI."""

    provider_name = "openai"

    def __init__(self, model: str = "gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY não encontrada. Verifique o arquivo .env.")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        max_output_tokens: int = 512,
    ) -> str:
        """Gera uma resposta utilizando a API da OpenAI."""

        start_time = perf_counter()
        prompt_length = len(user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                max_completion_tokens=max_output_tokens,
                temperature=0.1,
            )

            content = response.choices[0].message.content

            response_text = content.strip() if isinstance(content, str) else ""

            if not response_text:
                raise LLMResponseError("A OpenAI retornou uma resposta vazia.")

            duration_ms = round(
                (perf_counter() - start_time) * 1000,
                2,
            )

            usage = getattr(response, "usage", None)

            prompt_tokens = getattr(
                usage,
                "prompt_tokens",
                None,
            )

            output_tokens = getattr(
                usage,
                "completion_tokens",
                None,
            )

            total_tokens = getattr(
                usage,
                "total_tokens",
                None,
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
                "Chamada à OpenAI concluída",
                extra={
                    "event": "openai_request_completed",
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
                "Falha na chamada à OpenAI",
                extra={
                    "event": "openai_request_failed",
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
                    "O limite de requisições da OpenAI foi atingido."
                ) from exc

            if isinstance(exc, (APITimeoutError, TimeoutError)) or status_code == 408:
                raise LLMTimeoutError(
                    "A comunicação com a OpenAI excedeu o tempo limite."
                ) from exc

            if isinstance(exc, APIConnectionError) or status_code in (
                500,
                502,
                503,
                504,
            ):
                raise LLMTransientProviderError(
                    "O serviço OpenAI está temporariamente indisponível."
                ) from exc

            if isinstance(exc, APIStatusError):
                raise LLMProviderError(
                    f"A API OpenAI retornou um erro HTTP {status_code}."
                ) from exc

            raise LLMProviderError(
                "Falha na comunicação com o provedor OpenAI."
            ) from exc
