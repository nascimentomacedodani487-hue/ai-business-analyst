"""Métricas agregadas e individuais das chamadas aos provedores de LLM.

Mantém métricas globais separadas por provedor, permite atribuir chamadas
a análises por meio de ContextVar, registra traces com retenção limitada e
estima custos por modelo usando o módulo ``pricing``.

A estimativa não substitui a fatura do provedor. Ela usa os tokens reportados
pelos adaptadores e as tarifas cadastradas em ``ai_analyst.llm.pricing``.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar, Token
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, floor, isfinite
from numbers import Integral, Real
from threading import Lock
from typing import Iterator
from uuid import uuid4

from ai_analyst.llm.pricing import (
    PRICING_AS_OF,
    PricingTier,
    estimate_token_cost,
)

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"

DEFAULT_TRACE_CAPACITY = 1_000
DEFAULT_LATENCY_SAMPLE_CAPACITY = 1_000
DEFAULT_TRACE_LIMIT = 100

DEFAULT_PRICING_TIERS: dict[str, PricingTier] = {
    "gemini": "free",
    "openai": "standard",
    "anthropic": "standard",
}

PROVIDER_MODELS = {
    "gemini": DEFAULT_MODEL,
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


# ==========================================================
# VALIDAÇÃO
# ==========================================================


def _validate_duration_ms(duration_ms: float) -> float:
    """Valida uma duração numérica, finita e não negativa."""

    if isinstance(duration_ms, bool) or not isinstance(duration_ms, Real):
        raise TypeError("duration_ms deve ser um número real.")

    try:
        normalized = float(duration_ms)
    except (OverflowError, ValueError) as exc:
        raise ValueError("duration_ms deve ser finito.") from exc

    if not isfinite(normalized):
        raise ValueError("duration_ms deve ser finito.")

    if normalized < 0:
        raise ValueError("duration_ms não pode ser negativo.")

    return normalized


def _validate_token_count(value: int | None, field_name: str) -> int | None:
    """Aceita inteiros não negativos ou None quando o token não foi reportado."""

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} deve ser um número inteiro ou None.")

    normalized = int(value)

    if normalized < 0:
        raise ValueError(f"{field_name} não pode ser negativo.")

    return normalized


def _validate_positive_integer(value: int, field_name: str) -> int:
    """Valida um inteiro positivo."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} deve ser um número inteiro.")

    normalized = int(value)

    if normalized < 1:
        raise ValueError(f"{field_name} deve ser maior ou igual a 1.")

    return normalized


def _validate_pricing_tier(
    value: str,
    field_name: str = "pricing_tier",
) -> PricingTier:
    """Valida uma faixa de cobrança aceita pelo estimador."""

    if value not in ("standard", "free"):
        raise ValueError(f"{field_name} deve ser 'standard' ou 'free'.")

    return value  # type: ignore[return-value]


# ==========================================================
# LATÊNCIA
# ==========================================================


def _calculate_percentile(
    sorted_samples: list[float],
    percentile: float,
) -> float:
    """Calcula um percentil usando interpolação linear."""

    if not sorted_samples:
        return 0.0

    if len(sorted_samples) == 1:
        return round(sorted_samples[0], 2)

    position = (len(sorted_samples) - 1) * percentile

    lower_index = floor(position)
    upper_index = ceil(position)

    lower_value = sorted_samples[lower_index]
    upper_value = sorted_samples[upper_index]

    interpolated = lower_value + (
        upper_value - lower_value
    ) * (position - lower_index)

    return round(interpolated, 2)


def _build_latency_distribution(
    latency_samples: deque[float],
) -> dict:
    """Calcula p50 e p95 sobre a janela de amostras mantida em memória."""

    sorted_samples = sorted(latency_samples)

    return {
        "p50_ms": _calculate_percentile(sorted_samples, 0.50),
        "p95_ms": _calculate_percentile(sorted_samples, 0.95),
        "sample_count": len(sorted_samples),
        "sample_capacity": latency_samples.maxlen,
    }


# ==========================================================
# PRECIFICAÇÃO
# ==========================================================


def get_pricing_metadata(provider: str) -> dict:
    """Mantém os metadados legados de cobrança sem inventar custo por uso.

    O custo calculado a partir dos tokens fica em ``token_cost_estimate``.
    ``cost_estimate_usd`` é preservado para compatibilidade com consumidores
    antigos que ainda esperam o campo original.
    """

    if provider == "gemini":
        return {
            "pricing_tier": "free",
            "cost_estimate_usd": 0.0,
        }

    return {
        "pricing_tier": "usage_based",
        "cost_estimate_usd": None,
    }


def _estimate_cost_for_call(
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    thinking_tokens: int | None,
    pricing_tier: PricingTier,
) -> dict[str, object] | None:
    """Calcula o custo de uma chamada quando os dados necessários existem.

    Não estima custo quando input ou output não foram reportados. Um contador
    igual a zero é válido e não é confundido com ausência de informação.
    """

    if input_tokens is None or output_tokens is None:
        return None

    return estimate_token_cost(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        pricing_tier=pricing_tier,
    )


def _new_model_cost_bucket(
    provider: str,
    model: str,
    pricing_tier: str,
) -> dict:
    """Cria o acumulador de custo para um modelo e faixa de cobrança."""

    return {
        "provider": provider,
        "model": model,
        "pricing_tier": pricing_tier,
        "requests_total": 0,
        "requests_estimated": 0,
        "requests_unestimated": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "estimated_cost_usd": 0.0,
        "pricing_context_tiers": {},
        "pricing_contexts": {},
        "pricing_source": None,
        "long_context_threshold_tokens": None,
        "pricing_as_of": PRICING_AS_OF.isoformat(),
    }


def _record_model_cost(
    model_costs: dict[str, dict],
    *,
    provider: str,
    model: str,
    pricing_tier: PricingTier,
    input_tokens: int | None,
    output_tokens: int | None,
    thinking_tokens: int | None,
    estimate: dict[str, object] | None,
) -> None:
    """Acumula uma estimativa sem misturar tarifas de modelos diferentes."""

    bucket_key = model
    existing_bucket = model_costs.get(bucket_key)

    # O mesmo modelo pode ser executado em faixas diferentes durante
    # testes controlados. Não misture custos free e standard.
    if (
        existing_bucket is not None
        and existing_bucket["pricing_tier"] != pricing_tier
    ):
        bucket_key = f"{model}::{pricing_tier}"

    bucket = model_costs.setdefault(
        bucket_key,
        _new_model_cost_bucket(
            provider,
            model,
            pricing_tier,
        ),
    )

    bucket["requests_total"] += 1

    if input_tokens is not None:
        bucket["input_tokens"] += input_tokens

    if output_tokens is not None:
        bucket["output_tokens"] += output_tokens

    if thinking_tokens is not None:
        bucket["thinking_tokens"] += thinking_tokens

    if estimate is None:
        bucket["requests_unestimated"] += 1
        return

    bucket["requests_estimated"] += 1
    bucket["estimated_cost_usd"] += float(
        estimate["estimated_cost_usd"]
    )

    context_tier = str(estimate["pricing_context_tier"])

    context_tiers = bucket["pricing_context_tiers"]
    context_tiers[context_tier] = (
        context_tiers.get(context_tier, 0) + 1
    )

    context = bucket["pricing_contexts"].setdefault(
        context_tier,
        {
            "requests": 0,
            "estimated_cost_usd": 0.0,
            "rates": estimate["rates"],
        },
    )

    context["requests"] += 1
    context["estimated_cost_usd"] += float(
        estimate["estimated_cost_usd"]
    )

    bucket["pricing_source"] = estimate["pricing_source"]

    bucket["long_context_threshold_tokens"] = estimate[
        "long_context_threshold_tokens"
    ]


def _build_token_cost_snapshot(
    model_costs: dict[str, dict],
) -> dict:
    """Produz o resumo financeiro agregado por modelo e por provedor."""

    models: dict[str, dict] = {}

    requests_estimated = 0
    requests_unestimated = 0
    total_estimated_cost = 0.0

    has_any_estimate = False

    for model_key, bucket in sorted(model_costs.items()):
        model_estimated = bucket["requests_estimated"]
        model_unestimated = bucket["requests_unestimated"]

        has_model_estimate = model_estimated > 0

        models[model_key] = {
            "provider": bucket["provider"],
            "model": bucket["model"],
            "pricing_tier": bucket["pricing_tier"],
            "requests_total": bucket["requests_total"],
            "requests_estimated": model_estimated,
            "requests_unestimated": model_unestimated,
            "input_tokens": bucket["input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "thinking_tokens": bucket["thinking_tokens"],
            "estimated_cost_usd": (
                round(bucket["estimated_cost_usd"], 12)
                if has_model_estimate
                else None
            ),
            "pricing_context_tiers": dict(
                bucket["pricing_context_tiers"]
            ),
            "pricing_contexts": {
                context_name: {
                    "requests": context["requests"],
                    "estimated_cost_usd": round(
                        context["estimated_cost_usd"],
                        12,
                    ),
                    "rates": dict(context["rates"]),
                }
                for context_name, context
                in bucket["pricing_contexts"].items()
            },
            "pricing_source": bucket["pricing_source"],
            "long_context_threshold_tokens": bucket[
                "long_context_threshold_tokens"
            ],
            "pricing_as_of": bucket["pricing_as_of"],
        }

        requests_estimated += model_estimated
        requests_unestimated += model_unestimated

        if has_model_estimate:
            has_any_estimate = True
            total_estimated_cost += bucket["estimated_cost_usd"]

    return {
        "currency": "USD",
        "pricing_as_of": PRICING_AS_OF.isoformat(),
        "requests_estimated": requests_estimated,
        "requests_unestimated": requests_unestimated,
        "is_complete": requests_unestimated == 0,
        "estimated_cost_usd": (
            round(total_estimated_cost, 12)
            if has_any_estimate
            else None
        ),
        "models": models,
    }


# ==========================================================
# MÉTRICAS INDIVIDUAIS POR ANÁLISE
# ==========================================================


@dataclass
class AnalysisLLMMetrics:
    """Métricas das chamadas de LLM associadas a uma análise."""

    analysis_id: str
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL

    # Mantido depois dos campos originais para preservar
    # a compatibilidade dos argumentos posicionais.
    request_id: str | None = None

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

    _latency_samples: deque[float] = field(
        default_factory=lambda: deque(
            maxlen=DEFAULT_LATENCY_SAMPLE_CAPACITY
        ),
        repr=False,
        compare=False,
    )

    _model_costs: dict[str, dict] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    _lock: Lock = field(
        default_factory=Lock,
        repr=False,
        compare=False,
    )

    def record_success(
        self,
        model: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        total_tokens: int | None = None,
        provider: str | None = None,
        pricing_tier: PricingTier | None = None,
    ) -> None:
        """Registra uma chamada bem-sucedida e sua estimativa individual."""

        duration_ms = _validate_duration_ms(duration_ms)

        prompt_tokens = _validate_token_count(
            prompt_tokens,
            "prompt_tokens",
        )

        output_tokens = _validate_token_count(
            output_tokens,
            "output_tokens",
        )

        thinking_tokens = _validate_token_count(
            thinking_tokens,
            "thinking_tokens",
        )

        total_tokens = _validate_token_count(
            total_tokens,
            "total_tokens",
        )

        effective_provider = provider or self.provider

        effective_tier = _validate_pricing_tier(
            pricing_tier
            or DEFAULT_PRICING_TIERS.get(
                effective_provider,
                "standard",
            )
        )

        estimate = _estimate_cost_for_call(
            provider=effective_provider,
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            pricing_tier=effective_tier,
        )

        with self._lock:
            self.provider = effective_provider

            self.requests_total += 1
            self.requests_successful += 1

            self.model = model

            self.total_latency_ms += duration_ms
            self._latency_samples.append(duration_ms)

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
                    prompt_tokens
                    + output_tokens
                    + (thinking_tokens or 0)
                )

            _record_model_cost(
                self._model_costs,
                provider=effective_provider,
                model=model,
                pricing_tier=effective_tier,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                estimate=estimate,
            )

    def record_failure(
        self,
        model: str,
        duration_ms: float,
        provider: str | None = None,
    ) -> None:
        """Registra uma chamada que terminou com falha."""

        duration_ms = _validate_duration_ms(duration_ms)

        with self._lock:
            if provider is not None:
                self.provider = provider

            self.requests_total += 1
            self.requests_failed += 1
            self.tokens_unreported += 1

            self.model = model

            self.total_latency_ms += duration_ms
            self._latency_samples.append(duration_ms)

    def record_retry(self) -> None:
        """Registra uma nova tentativa."""

        with self._lock:
            self.retries += 1

    def snapshot(self) -> dict:
        """Retorna uma cópia consistente das métricas da análise."""

        with self._lock:
            average_latency = (
                round(
                    self.total_latency_ms / self.requests_total,
                    2,
                )
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
                    "total_ms": round(
                        self.total_latency_ms,
                        2,
                    ),
                    "average_ms": average_latency,
                },
                "latency_distribution": _build_latency_distribution(
                    self._latency_samples
                ),
                "token_cost_estimate": _build_token_cost_snapshot(
                    self._model_costs
                ),
            }


# ==========================================================
# GERENCIADOR GLOBAL DE MÉTRICAS
# ==========================================================


class LLMMetrics:
    """Gerencia métricas globais, individuais e traces dos provedores."""

    def __init__(
        self,
        max_trace_events: int = DEFAULT_TRACE_CAPACITY,
        pricing_tiers: dict[str, PricingTier] | None = None,
    ) -> None:
        self._max_trace_events = _validate_positive_integer(
            max_trace_events,
            "max_trace_events",
        )

        self._pricing_tiers = dict(DEFAULT_PRICING_TIERS)

        if pricing_tiers is not None:
            for provider, tier in pricing_tiers.items():
                self._pricing_tiers[provider] = _validate_pricing_tier(
                    tier,
                    f"pricing_tiers[{provider!r}]",
                )

        self._lock = Lock()

        self._active_analysis: ContextVar[
            AnalysisLLMMetrics | None
        ] = ContextVar(
            f"active_llm_analysis_{id(self)}",
            default=None,
        )

        self._provider_metrics: dict[str, dict] = {}

        self._trace_events: deque[dict] = deque(
            maxlen=self._max_trace_events
        )

        self.reset()

    def _get_pricing_tier(
        self,
        provider: str,
        pricing_tier: PricingTier | None,
    ) -> PricingTier:
        """Resolve a faixa de cobrança efetiva."""

        if pricing_tier is not None:
            return _validate_pricing_tier(pricing_tier)

        return self._pricing_tiers.get(
            provider,
            "standard",
        )

    def _get_provider_metrics(self, provider: str) -> dict:
        """Obtém ou inicializa o acumulador de um provedor."""

        if provider not in self._provider_metrics:
            self._provider_metrics[provider] = {
                "model": PROVIDER_MODELS.get(
                    provider,
                    "unknown",
                ),
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
                "latency_samples": deque(
                    maxlen=DEFAULT_LATENCY_SAMPLE_CAPACITY
                ),
                "model_costs": {},
            }

        return self._provider_metrics[provider]

    # ------------------------------------------------------
    # CONTEXTO INDIVIDUAL
    # ------------------------------------------------------

    @contextmanager
    def track_analysis(
        self,
        analysis_id: str,
        request_id: str | None = None,
    ) -> Iterator[AnalysisLLMMetrics]:
        """Atribui as chamadas feitas neste contexto à análise informada."""

        analysis_metrics = AnalysisLLMMetrics(
            analysis_id=analysis_id,
            request_id=request_id,
        )

        token: Token = self._active_analysis.set(
            analysis_metrics
        )

        try:
            yield analysis_metrics
        finally:
            self._active_analysis.reset(token)

    # ------------------------------------------------------
    # TRACING
    # ------------------------------------------------------

    def _record_trace(
        self,
        event_type: str,
        status: str,
        provider: str,
        model: str,
        duration_ms: float | None = None,
        tokens: dict | None = None,
        retry: dict | None = None,
        error_type: str | None = None,
    ) -> None:
        """Armazena evento sem registrar prompts ou respostas."""

        analysis_metrics = self._active_analysis.get()

        event = {
            "event_id": str(uuid4()),
            "event_type": event_type,
            "status": status,
            "timestamp": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "analysis_id": (
                analysis_metrics.analysis_id
                if analysis_metrics
                else None
            ),
            "request_id": (
                analysis_metrics.request_id
                if analysis_metrics
                else None
            ),
            "provider": provider,
            "model": model,
            "duration_ms": (
                round(duration_ms, 2)
                if duration_ms is not None
                else None
            ),
            "tokens": tokens,
            "retry": retry,
            "error_type": error_type,
        }

        with self._lock:
            self._trace_events.append(event)

    # ------------------------------------------------------
    # SUCESSO
    # ------------------------------------------------------

    def record_success(
        self,
        model: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        total_tokens: int | None = None,
        provider: str = DEFAULT_PROVIDER,
        pricing_tier: PricingTier | None = None,
    ) -> None:
        """Registra sucesso, tokens e custo estimado para o modelo exato."""

        duration_ms = _validate_duration_ms(duration_ms)

        prompt_tokens = _validate_token_count(
            prompt_tokens,
            "prompt_tokens",
        )

        output_tokens = _validate_token_count(
            output_tokens,
            "output_tokens",
        )

        thinking_tokens = _validate_token_count(
            thinking_tokens,
            "thinking_tokens",
        )

        total_tokens = _validate_token_count(
            total_tokens,
            "total_tokens",
        )

        effective_tier = self._get_pricing_tier(
            provider,
            pricing_tier,
        )

        effective_total_tokens = total_tokens

        if (
            effective_total_tokens is None
            and prompt_tokens is not None
            and output_tokens is not None
        ):
            effective_total_tokens = (
                prompt_tokens
                + output_tokens
                + (thinking_tokens or 0)
            )

        estimate = _estimate_cost_for_call(
            provider=provider,
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            pricing_tier=effective_tier,
        )

        with self._lock:
            current = self._get_provider_metrics(provider)

            current["requests_total"] += 1
            current["requests_successful"] += 1
            current["model"] = model

            current["total_latency_ms"] += duration_ms
            current["latency_samples"].append(duration_ms)

            if prompt_tokens is None:
                current["tokens_unreported"] += 1
            else:
                current["input_tokens"] += prompt_tokens

            if output_tokens is not None:
                current["output_tokens"] += output_tokens

            if thinking_tokens is not None:
                current["thinking_tokens"] += thinking_tokens

            if effective_total_tokens is not None:
                current["total_tokens"] += effective_total_tokens

            _record_model_cost(
                current["model_costs"],
                provider=provider,
                model=model,
                pricing_tier=effective_tier,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                estimate=estimate,
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
                pricing_tier=effective_tier,
            )

        self._record_trace(
            event_type="llm_call",
            status="success",
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            tokens={
                "input": prompt_tokens,
                "output": output_tokens,
                "thinking": thinking_tokens,
                "total": effective_total_tokens,
            },
        )

    # ------------------------------------------------------
    # FALHA
    # ------------------------------------------------------

    def record_failure(
        self,
        model: str,
        duration_ms: float,
        provider: str = DEFAULT_PROVIDER,
        error_type: str | None = None,
    ) -> None:
        """Registra uma chamada que terminou com falha."""

        duration_ms = _validate_duration_ms(duration_ms)

        with self._lock:
            current = self._get_provider_metrics(provider)

            current["requests_total"] += 1
            current["requests_failed"] += 1
            current["tokens_unreported"] += 1
            current["model"] = model

            current["total_latency_ms"] += duration_ms
            current["latency_samples"].append(duration_ms)

        analysis_metrics = self._active_analysis.get()

        if analysis_metrics is not None:
            analysis_metrics.record_failure(
                model=model,
                duration_ms=duration_ms,
                provider=provider,
            )

        self._record_trace(
            event_type="llm_call",
            status="failure",
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            error_type=error_type,
        )

    # ------------------------------------------------------
    # RETRY
    # ------------------------------------------------------

    def record_retry(
        self,
        provider: str = DEFAULT_PROVIDER,
        *,
        model: str | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
        delay_ms: float | None = None,
        error_type: str | None = None,
    ) -> None:
        """Registra uma nova tentativa para o provedor correspondente."""

        if delay_ms is not None:
            delay_ms = _validate_duration_ms(delay_ms)

        with self._lock:
            current = self._get_provider_metrics(provider)
            current["retries"] += 1

        analysis_metrics = self._active_analysis.get()

        if analysis_metrics is not None:
            analysis_metrics.record_retry()

        self._record_trace(
            event_type="llm_retry",
            status="scheduled",
            provider=provider,
            model=model or PROVIDER_MODELS.get(
                provider,
                "unknown",
            ),
            retry={
                "attempt": attempt,
                "next_attempt": (
                    attempt + 1
                    if attempt is not None
                    else None
                ),
                "max_attempts": max_attempts,
                "delay_ms": delay_ms,
            },
            error_type=error_type,
        )

    # ------------------------------------------------------
    # SNAPSHOTS
    # ------------------------------------------------------

    def snapshot(self, provider: str = DEFAULT_PROVIDER) -> dict:
        """Retorna métricas agregadas de um provedor específico."""

        with self._lock:
            current = self._get_provider_metrics(provider)

            requests_total = current["requests_total"]

            average_latency = (
                round(
                    current["total_latency_ms"] / requests_total,
                    2,
                )
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
                # Estrutura legada preservada.
                "latency": {
                    "average_ms": average_latency,
                },
                "latency_distribution": _build_latency_distribution(
                    current["latency_samples"]
                ),
                "token_cost_estimate": _build_token_cost_snapshot(
                    current["model_costs"]
                ),
            }

            result.update(
                get_pricing_metadata(provider)
            )

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
        """Retorna métricas agregadas de todos os provedores registrados."""

        with self._lock:
            providers = list(self._provider_metrics)

        return {
            provider: self.snapshot(provider=provider)
            for provider in providers
        }

    # ------------------------------------------------------
    # CONSULTA DOS TRACES
    # ------------------------------------------------------

    def get_recent_traces(
        self,
        *,
        provider: str | None = None,
        analysis_id: str | None = None,
        request_id: str | None = None,
        status: str | None = None,
        limit: int = DEFAULT_TRACE_LIMIT,
    ) -> list[dict]:
        """Retorna eventos recentes, com filtros e cópias defensivas."""

        normalized_limit = min(
            _validate_positive_integer(
                limit,
                "limit",
            ),
            self._max_trace_events,
        )

        results = []

        with self._lock:
            for event in reversed(self._trace_events):
                if provider is not None and event["provider"] != provider:
                    continue

                if (
                    analysis_id is not None
                    and event["analysis_id"] != analysis_id
                ):
                    continue

                if (
                    request_id is not None
                    and event["request_id"] != request_id
                ):
                    continue

                if status is not None and event["status"] != status:
                    continue

                results.append(deepcopy(event))

                if len(results) >= normalized_limit:
                    break

        return results

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------

    def reset(self) -> None:
        """Zera métricas globais e remove os traces armazenados."""

        with self._lock:
            self._provider_metrics = {}
            self._trace_events.clear()


llm_metrics = LLMMetrics()