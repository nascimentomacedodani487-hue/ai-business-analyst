
"""
Integração com a API Google Gemini.
"""

import logging
import os
from time import perf_counter

from dotenv import load_dotenv
from google import genai
from google.genai import types

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


class GeminiClient:
    """Gerencia a comunicação com o modelo Gemini."""

    provider_name = "gemini"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. "
                "Verifique o arquivo .env."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate_response(
        self,
        user_prompt: str,
        system_prompt: str,
        max_output_tokens: int = 512,
    ) -> str:
        """
        Gera uma resposta utilizando instruções de sistema.

        Registra duração, utilização de tokens e falhas.
        Converte falhas específicas em exceções padronizadas.
        """

        start_time = perf_counter()
        prompt_length = len(user_prompt)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                ),
            )

            response_text = (
                response.text.strip()
                if response.text
                else ""
            )

            if not response_text:
                raise LLMResponseError(
                    "O Gemini retornou uma resposta vazia."
                )

            duration_ms = round(
                (perf_counter() - start_time) * 1000,
                2,
            )

            usage = getattr(response, "usage_metadata", None)

            prompt_tokens = getattr(
                usage,
                "prompt_token_count",
                None,
            )

            output_tokens = getattr(
                usage,
                "candidates_token_count",
                None,
            )

            thinking_tokens = getattr(
                usage,
                "thoughts_token_count",
                None,
            )

            total_tokens = getattr(
                usage,
                "total_token_count",
                None,
            )

            llm_metrics.record_success(
                provider=self.provider_name,
                model=self.model,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                total_tokens=total_tokens,
            )

            logger.info(
                "Chamada ao Gemini concluída",
                extra={
                    "event": "gemini_request_completed",
                    "provider": self.provider_name,
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "prompt_length": prompt_length,
                    "response_length": len(response_text),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "total_tokens": total_tokens,
                    "max_output_tokens": max_output_tokens,
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
                "Falha na chamada ao Gemini",
                extra={
                    "event": "gemini_request_failed",
                    "provider": self.provider_name,
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "prompt_length": prompt_length,
                },
            )

            if isinstance(exc, LLMError):
                raise

            status_code = (
                getattr(exc, "code", None)
                or getattr(exc, "status_code", None)
            )

            if status_code == 429:
                raise LLMRateLimitError(
                    "O limite de requisições ou a cota do Gemini "
                    "foi atingido."
                ) from exc

            if status_code == 408:
                raise LLMTimeoutError(
                    "A comunicação com o Gemini excedeu "
                    "o tempo limite."
                ) from exc

            if status_code in (500, 502, 503, 504):
                raise LLMTransientProviderError(
                    "O serviço Gemini está temporariamente "
                    "indisponível."
                ) from exc

            if isinstance(exc, TimeoutError):
                raise LLMTimeoutError(
                    "A comunicação com o Gemini excedeu "
                    "o tempo limite."
                ) from exc

            raise LLMProviderError(
                "Falha na comunicação com o provedor Gemini."
            ) from exc
