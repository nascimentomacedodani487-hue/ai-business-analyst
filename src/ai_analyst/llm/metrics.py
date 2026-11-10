"""
Métricas agregadas e individuais das chamadas aos provedores de LLM.

Mantém métricas globais separadas por provedor e permite atribuir
chamadas a uma análise específica por meio de ContextVar.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"

PROVIDER_MODELS = {
    "gemini": DEFAULT_MODEL,
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


def get_pricing_metadata(provider: str) -> dict:
    """Retorna informações de cobrança sem inventar custos."""

    if provider == "gemini":
        return {
            "pricing_tier": "free",
            "cost_estimate_usd": 0.0,
        }

    return {
        "pricing_tier": "usage_based",
        "cost_estimate_usd": None,
    }


@dataclass
class AnalysisLLMMetrics:
    """Métricas das chamadas de LLM associadas a uma análise."""

    analysis_id: str
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL

    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    retries: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    tokens_unreported: int = 0

    total_latency_ms: float = 0.0

    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record_success(
        self,
        model: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        total_tokens: int | None = None,
        provider: str | None = None,
    ) -> None:
        with self._lock:
            if provider is not None:
                self.provider = provider

            self.requests_total += 1
            self.requests_successful += 1
            self.model = model
            self.total_latency_ms += duration_ms

            if prompt_tokens is None:
                self.tokens_unreported += 1
            else:
                self.input_tokens += prompt_tokens

            if output_tokens is not None:
                self.output_tokens += output_tokens

            if thinking_tokens is not None:
                self.thinking_tokens += thinking_tokens

            if total_tokens is not None:
                self.total_tokens += total_tokens
            elif prompt_tokens is not None and output_tokens is not None:
                self.total_tokens += (
                    prompt_tokens + output_tokens + (thinking_tokens or 0)
                )

    def record_failure(
        self,
        model: str,
        duration_ms: float,
        provider: str | None = None,
    ) -> None:
        with self._lock:
            if provider is not None:
                self.provider = provider

            self.requests_total += 1
            self.requests_failed += 1
            self.tokens_unreported += 1
            self.model = model
            self.total_latency_ms += duration_ms

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def snapshot(self) -> dict:
        with self._lock:
            average_latency = (
                round(self.total_latency_ms / self.requests_total, 2)
                if self.requests_total
                else 0.0
            )

            return {
                "analysis_id": self.analysis_id,
                "provider": self.provider,
                "model": self.model,
                "requests": {
                    "total": self.requests_total,
                    "successful": self.requests_successful,
                    "failed": self.requests_failed,
                    "retries": self.retries,
                },
                "tokens": {
                    "input": self.input_tokens,
                    "output": self.output_tokens,
                    "thinking": self.thinking_tokens,
                    "total": self.total_tokens,
                    "unreported": self.tokens_unreported,
                },
                "latency": {
                    "total_ms": round(self.total_latency_ms, 2),
                    "average_ms": average_latency,
                },
            }


class LLMMetrics:
    """Gerencia métricas globais e individuais por análise."""

    def __init__(self):
        self._lock = Lock()

        self._active_analysis: ContextVar[AnalysisLLMMetrics | None] = ContextVar(
            "active_llm_analysis", default=None
        )

        self._provider_metrics: dict[str, dict] = {}
        self.reset()

    def _get_provider_metrics(self, provider: str) -> dict:
        """Obtém ou inicializa o acumulador de um provedor."""

        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = {
                "model": PROVIDER_MODELS.get(provider, "unknown"),
                "requests_total": 0,
                "requests_successful": 0,
                "requests_failed": 0,
                "retries": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "total_tokens": 0,
                "tokens_unreported": 0,
                "total_latency_ms": 0.0,
            }

        return self._provider_metrics[provider]

    @contextmanager
    def track_analysis(
        self,
        analysis_id: str,
    ) -> Iterator[AnalysisLLMMetrics]:
        """Associa chamadas de LLM do contexto atual a uma análise."""

        analysis_metrics = AnalysisLLMMetrics(
            analysis_id=analysis_id,
        )

        token: Token = self._active_analysis.set(analysis_metrics)

        try:
            yield analysis_metrics
        finally:
            self._active_analysis.reset(token)

    def record_success(
        self,
        model: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        total_tokens: int | None = None,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        with self._lock:
            current = self._get_provider_metrics(provider)

            current["requests_total"] += 1
            current["requests_successful"] += 1
            current["model"] = model
            current["total_latency_ms"] += duration_ms

            if prompt_tokens is None:
                current["tokens_unreported"] += 1
            else:
                current["input_tokens"] += prompt_tokens

            if output_tokens is not None:
                current["output_tokens"] += output_tokens

            if thinking_tokens is not None:
                current["thinking_tokens"] += thinking_tokens

            if total_tokens is not None:
                current["total_tokens"] += total_tokens
            elif prompt_tokens is not None and output_tokens is not None:
                current["total_tokens"] += (
                    prompt_tokens + output_tokens + (thinking_tokens or 0)
                )

        analysis_metrics = self._active_analysis.get()

        if analysis_metrics is not None:
            analysis_metrics.record_success(
                model=model,
                duration_ms=duration_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                total_tokens=total_tokens,
                provider=provider,
            )

    def record_failure(
        self,
        model: str,
        duration_ms: float,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        with self._lock:
            current = self._get_provider_metrics(provider)

            current["requests_total"] += 1
            current["requests_failed"] += 1
            current["tokens_unreported"] += 1
            current["model"] = model
            current["total_latency_ms"] += duration_ms

        analysis_metrics = self._active_analysis.get()

        if analysis_metrics is not None:
            analysis_metrics.record_failure(
                model=model,
                duration_ms=duration_ms,
                provider=provider,
            )

    def record_retry(
        self,
        provider: str = DEFAULT_PROVIDER,
    ) -> None:
        with self._lock:
            current = self._get_provider_metrics(provider)
            current["retries"] += 1

        analysis_metrics = self._active_analysis.get()

        if analysis_metrics is not None:
            analysis_metrics.record_retry()

    def snapshot(self, provider: str = DEFAULT_PROVIDER) -> dict:
        """Retorna métricas de um provedor específico."""

        with self._lock:
            current = self._get_provider_metrics(provider)

            requests_total = current["requests_total"]

            average_latency = (
                round(current["total_latency_ms"] / requests_total, 2)
                if requests_total
                else 0.0
            )

            result = {
                "provider": provider,
                "model": current["model"],
                "requests": {
                    "total": requests_total,
                    "successful": current["requests_successful"],
                    "failed": current["requests_failed"],
                    "retries": current["retries"],
                },
                "tokens": {
                    "input": current["input_tokens"],
                    "output": current["output_tokens"],
                    "thinking": current["thinking_tokens"],
                    "total": current["total_tokens"],
                    "unreported": current["tokens_unreported"],
                },
                "latency": {
                    "average_ms": average_latency,
                },
            }

            result.update(get_pricing_metadata(provider))

            if provider == "gemini":
                result["quota"] = {
                    "status": "not_exposed_by_api",
                    "note": (
                        "A API não disponibiliza neste painel "
                        "o saldo exato da cota gratuita."
                    ),
                }

            return result

    def snapshot_all(self) -> dict[str, dict]:
        """Retorna métricas de todos os provedores registrados."""

        with self._lock:
            providers = list(self._provider_metrics)

        return {provider: self.snapshot(provider=provider) for provider in providers}

    def reset(self) -> None:
        """Zera todas as métricas globais."""

        with self._lock:
            self._provider_metrics = {}


llm_metrics = LLMMetrics()
