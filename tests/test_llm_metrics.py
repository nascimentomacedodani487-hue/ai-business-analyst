"""Testes das métricas globais, por análise e da estimativa de custos."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from ai_analyst.llm.metrics import (
    AnalysisLLMMetrics,
    LLMMetrics,
    get_pricing_metadata,
)


@pytest.fixture
def metrics() -> LLMMetrics:
    return LLMMetrics()


def _assert_empty_snapshot(snapshot: dict) -> None:
    assert snapshot["requests"] == {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "retries": 0,
    }
    assert snapshot["tokens"] == {
        "input": 0,
        "output": 0,
        "thinking": 0,
        "total": 0,
        "unreported": 0,
    }
    assert snapshot["latency"]["average_ms"] == 0.0
    assert snapshot["token_cost_estimate"]["estimated_cost_usd"] is None
    assert snapshot["token_cost_estimate"]["requests_estimated"] == 0
    assert snapshot["token_cost_estimate"]["requests_unestimated"] == 0


# ============================================================
# ESTADO INICIAL E OPERAÇÕES BÁSICAS
# ============================================================


def test_metrics_initial_state(metrics: LLMMetrics) -> None:
    _assert_empty_snapshot(metrics.snapshot())


def test_record_success(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=120.0,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=10,
        total_tokens=160,
    )

    snapshot = metrics.snapshot()

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["successful"] == 1

    assert snapshot["tokens"] == {
        "input": 100,
        "output": 50,
        "thinking": 10,
        "total": 160,
        "unreported": 0,
    }

    assert snapshot["token_cost_estimate"]["estimated_cost_usd"] == 0.0
    assert snapshot["token_cost_estimate"]["is_complete"] is True


def test_record_failure(metrics: LLMMetrics) -> None:
    metrics.record_failure(
        model="gemini-2.5-flash",
        duration_ms=50.0,
        error_type="TimeoutError",
    )

    snapshot = metrics.snapshot()

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["failed"] == 1
    assert snapshot["tokens"]["unreported"] == 1
    assert snapshot["token_cost_estimate"]["requests_unestimated"] == 0


def test_record_retry(metrics: LLMMetrics) -> None:
    metrics.record_retry(
        model="gemini-2.5-flash",
        attempt=1,
        max_attempts=3,
        delay_ms=250,
        error_type="RateLimitError",
    )

    snapshot = metrics.snapshot()

    assert snapshot["requests"]["retries"] == 1

    trace = metrics.get_recent_traces()[0]

    assert trace["event_type"] == "llm_retry"
    assert trace["retry"]["next_attempt"] == 2


def test_multiple_requests_aggregate_metrics(metrics: LLMMetrics) -> None:
    for _ in range(3):
        metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=100.0,
            prompt_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

    snapshot = metrics.snapshot()

    assert snapshot["requests"]["total"] == 3
    assert snapshot["tokens"]["input"] == 30
    assert snapshot["tokens"]["output"] == 15
    assert snapshot["tokens"]["total"] == 45
    assert snapshot["latency"]["average_ms"] == 100.0
    assert snapshot["token_cost_estimate"]["requests_estimated"] == 3


# ============================================================
# CONTADORES DE TOKENS
# ============================================================


def test_missing_token_usage_is_not_invented(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=10.0,
        prompt_tokens=None,
        output_tokens=20,
    )

    snapshot = metrics.snapshot()

    assert snapshot["tokens"]["input"] == 0
    assert snapshot["tokens"]["output"] == 20
    assert snapshot["tokens"]["unreported"] == 1

    cost = snapshot["token_cost_estimate"]

    assert cost["estimated_cost_usd"] is None
    assert cost["requests_estimated"] == 0
    assert cost["requests_unestimated"] == 1
    assert cost["is_complete"] is False


def test_total_tokens_fallback(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=10.0,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=10,
        total_tokens=None,
    )

    assert metrics.snapshot()["tokens"]["total"] == 160


def test_zero_token_counts_are_valid(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=0.0,
        prompt_tokens=0,
        output_tokens=0,
        thinking_tokens=0,
        total_tokens=0,
    )

    snapshot = metrics.snapshot()

    assert snapshot["tokens"]["unreported"] == 0
    assert snapshot["token_cost_estimate"]["requests_estimated"] == 1
    assert snapshot["token_cost_estimate"]["estimated_cost_usd"] == 0.0


def test_reset_clears_metrics(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=1,
        prompt_tokens=10,
        output_tokens=10,
    )

    metrics.record_retry()
    metrics.reset()

    _assert_empty_snapshot(metrics.snapshot())

    assert metrics.get_recent_traces() == []


# ============================================================
# PRECIFICAÇÃO E ESTIMATIVA DE CUSTOS
# ============================================================


def test_free_tier_does_not_report_paid_cost(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=10,
        prompt_tokens=100_000,
        output_tokens=50_000,
        thinking_tokens=10_000,
    )

    cost = metrics.snapshot()["token_cost_estimate"]

    assert cost["estimated_cost_usd"] == 0.0

    assert cost["models"]["gemini-2.5-flash"]["pricing_tier"] == "free"


def test_record_success_with_openai_provider(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gpt-4.1-mini",
        duration_ms=100,
        prompt_tokens=1_000,
        output_tokens=500,
        total_tokens=1_500,
        provider="openai",
    )

    snapshot = metrics.snapshot("openai")

    assert snapshot["pricing_tier"] == "usage_based"

    # Campo legado preservado.
    assert snapshot["cost_estimate_usd"] is None

    assert snapshot["token_cost_estimate"]["estimated_cost_usd"] == (
        pytest.approx(0.0012)
    )


def test_record_success_with_anthropic_provider(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="claude-haiku-4-5-20251001",
        duration_ms=100,
        prompt_tokens=1_000,
        output_tokens=500,
        total_tokens=1_500,
        provider="anthropic",
    )

    snapshot = metrics.snapshot("anthropic")

    assert snapshot["token_cost_estimate"]["estimated_cost_usd"] == (
        pytest.approx(0.0035)
    )


def test_providers_have_independent_metrics(metrics: LLMMetrics) -> None:
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=10,
        prompt_tokens=10,
        output_tokens=5,
        provider="gemini",
    )

    metrics.record_success(
        model="gpt-4.1-mini",
        duration_ms=20,
        prompt_tokens=20,
        output_tokens=10,
        provider="openai",
    )

    assert metrics.snapshot("gemini")["requests"]["total"] == 1
    assert metrics.snapshot("openai")["requests"]["total"] == 1

    assert set(metrics.snapshot_all()) == {
        "gemini",
        "openai",
    }


def test_analysis_metrics_preserve_provider(metrics: LLMMetrics) -> None:
    with metrics.track_analysis("analysis-001") as analysis:
        metrics.record_success(
            model="gpt-4.1-mini",
            duration_ms=15,
            prompt_tokens=100,
            output_tokens=50,
            provider="openai",
        )

        snapshot = analysis.snapshot()

    assert snapshot["analysis_id"] == "analysis-001"
    assert snapshot["provider"] == "openai"

    assert snapshot["token_cost_estimate"]["estimated_cost_usd"] == (
        pytest.approx(0.00012)
    )


def test_same_provider_aggregates_cost_by_model_not_last_model(
    metrics: LLMMetrics,
) -> None:
    metrics.record_success(
        model="gpt-4.1-mini",
        duration_ms=1,
        prompt_tokens=1_000,
        output_tokens=500,
        provider="openai",
    )

    metrics.record_success(
        model="gpt-6-luna",
        duration_ms=1,
        prompt_tokens=1_000,
        output_tokens=500,
        provider="openai",
    )

    cost = metrics.snapshot("openai")["token_cost_estimate"]

    assert set(cost["models"]) == {
        "gpt-4.1-mini",
        "gpt-6-luna",
    }

    assert cost["models"]["gpt-4.1-mini"]["estimated_cost_usd"] == (
        pytest.approx(0.0012)
    )

    assert cost["models"]["gpt-6-luna"]["estimated_cost_usd"] == (
        pytest.approx(0.00035)
    )

    assert cost["estimated_cost_usd"] == pytest.approx(0.00155)


def test_same_model_keeps_separate_costs_for_different_pricing_tiers() -> None:
    metrics = LLMMetrics()

    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=1,
        prompt_tokens=100,
        output_tokens=50,
        pricing_tier="free",
    )

    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=1,
        prompt_tokens=100,
        output_tokens=50,
        pricing_tier="standard",
    )

    models = metrics.snapshot()["token_cost_estimate"]["models"]

    assert models["gemini-2.5-flash"]["pricing_tier"] == "free"

    assert models["gemini-2.5-flash::standard"]["pricing_tier"] == "standard"

    assert metrics.snapshot()["token_cost_estimate"]["estimated_cost_usd"] == (
        pytest.approx(0.000155)
    )


def test_gemini_thinking_tokens_are_billed_as_output_when_paid() -> None:
    paid_metrics = LLMMetrics(pricing_tiers={"gemini": "standard"})

    paid_metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=1,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=10,
    )

    cost = paid_metrics.snapshot()["token_cost_estimate"]

    model_cost = cost["models"]["gemini-2.5-flash"]["estimated_cost_usd"]

    # 100 * 0.30/1e6 + (50 + 10) * 2.50/1e6
    assert model_cost == pytest.approx(0.00018)


def test_gpt6_long_context_uses_long_context_rates(
    metrics: LLMMetrics,
) -> None:
    metrics.record_success(
        model="gpt-6-luna",
        duration_ms=1,
        prompt_tokens=272_001,
        output_tokens=1_000,
        provider="openai",
    )

    model_cost = metrics.snapshot("openai")["token_cost_estimate"]["models"][
        "gpt-6-luna"
    ]

    assert model_cost["estimated_cost_usd"] == pytest.approx(0.0551502)

    assert model_cost["pricing_context_tiers"] == {
        "long_context": 1,
    }

    assert model_cost["pricing_source"].endswith("/models/gpt-6-luna")

    assert model_cost["pricing_contexts"]["long_context"]["rates"] == {
        "input_usd_per_million": 0.2,
        "output_usd_per_million": 0.75,
    }


def test_haiku55_long_context_uses_long_context_rates(
    metrics: LLMMetrics,
) -> None:
    metrics.record_success(
        model="claude-haiku-5-5",
        duration_ms=1,
        prompt_tokens=100_001,
        output_tokens=1_000,
        provider="anthropic",
    )

    model_cost = metrics.snapshot("anthropic")["token_cost_estimate"]["models"][
        "claude-haiku-5-5"
    ]

    assert model_cost["estimated_cost_usd"] == pytest.approx(0.0525005)

    assert model_cost["pricing_context_tiers"] == {
        "long_context": 1,
    }


def test_unknown_model_is_not_given_an_invented_cost(
    metrics: LLMMetrics,
) -> None:
    metrics.record_success(
        model="model-not-in-pricing-table",
        duration_ms=1,
        prompt_tokens=100,
        output_tokens=50,
        provider="openai",
    )

    cost = metrics.snapshot("openai")["token_cost_estimate"]

    assert cost["estimated_cost_usd"] is None
    assert cost["requests_unestimated"] == 1
    assert cost["is_complete"] is False


def test_get_pricing_metadata_keeps_legacy_contract() -> None:
    assert get_pricing_metadata("gemini") == {
        "pricing_tier": "free",
        "cost_estimate_usd": 0.0,
    }

    assert get_pricing_metadata("openai") == {
        "pricing_tier": "usage_based",
        "cost_estimate_usd": None,
    }


# ============================================================
# VALIDAÇÃO DE FAIXAS DE PREÇO
# ============================================================


@pytest.mark.parametrize(
    "invalid_tier",
    [
        "premium",
        "",
        None,
        123,
    ],
)
def test_invalid_pricing_tier_is_rejected(
    invalid_tier: object,
) -> None:
    with pytest.raises(ValueError):
        LLMMetrics(
            pricing_tiers={"gemini": invalid_tier}  # type: ignore[dict-item]
        )


# ============================================================
# VALIDAÇÃO DE DURAÇÃO
# ============================================================


@pytest.mark.parametrize(
    "invalid_duration",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "10",
        None,
        10**400,
    ],
)
def test_record_success_rejects_invalid_duration_without_mutating_metrics(
    metrics: LLMMetrics,
    invalid_duration: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=invalid_duration,  # type: ignore[arg-type]
            prompt_tokens=1,
            output_tokens=1,
        )

    _assert_empty_snapshot(metrics.snapshot())


@pytest.mark.parametrize(
    "invalid_duration",
    [
        -1,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "10",
        None,
    ],
)
def test_record_failure_rejects_invalid_duration_without_mutating_metrics(
    metrics: LLMMetrics,
    invalid_duration: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        metrics.record_failure(
            model="gemini-2.5-flash",
            duration_ms=invalid_duration,  # type: ignore[arg-type]
        )

    _assert_empty_snapshot(metrics.snapshot())


# ============================================================
# VALIDAÇÃO DOS CONTADORES DE TOKENS
# ============================================================


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt_tokens", -1, ValueError),
        ("prompt_tokens", 1.5, TypeError),
        ("prompt_tokens", True, TypeError),
        ("output_tokens", -1, ValueError),
        ("output_tokens", 1.5, TypeError),
        ("output_tokens", True, TypeError),
        ("thinking_tokens", -1, ValueError),
        ("thinking_tokens", 1.5, TypeError),
        ("thinking_tokens", True, TypeError),
        ("total_tokens", -1, ValueError),
        ("total_tokens", 1.5, TypeError),
        ("total_tokens", True, TypeError),
    ],
)
def test_record_success_rejects_invalid_token_counts_without_mutating_metrics(
    metrics: LLMMetrics,
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    arguments = {
        "model": "gemini-2.5-flash",
        "duration_ms": 1,
        "prompt_tokens": 1,
        "output_tokens": 1,
        "thinking_tokens": 0,
        "total_tokens": 2,
    }

    arguments[field] = value

    with pytest.raises(error):
        metrics.record_success(**arguments)  # type: ignore[arg-type]

    _assert_empty_snapshot(metrics.snapshot())


@pytest.mark.parametrize(
    ("invalid_duration"),
    [
        -1,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_analysis_metrics_reject_invalid_duration_without_mutating_state(
    invalid_duration: object,
) -> None:
    analysis = AnalysisLLMMetrics(analysis_id="invalid-duration")

    with pytest.raises((TypeError, ValueError)):
        analysis.record_success(
            model="gemini-2.5-flash",
            duration_ms=invalid_duration,  # type: ignore[arg-type]
            prompt_tokens=1,
            output_tokens=1,
        )

    snapshot = analysis.snapshot()

    assert snapshot["requests"]["total"] == 0
    assert snapshot["tokens"]["total"] == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt_tokens", -1, ValueError),
        ("prompt_tokens", 1.5, TypeError),
        ("prompt_tokens", True, TypeError),
        ("output_tokens", -1, ValueError),
        ("output_tokens", 1.5, TypeError),
        ("output_tokens", True, TypeError),
        ("thinking_tokens", -1, ValueError),
        ("thinking_tokens", 1.5, TypeError),
        ("thinking_tokens", True, TypeError),
        ("total_tokens", -1, ValueError),
        ("total_tokens", 1.5, TypeError),
        ("total_tokens", True, TypeError),
    ],
)
def test_analysis_metrics_reject_invalid_token_counts_without_mutating_state(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    analysis = AnalysisLLMMetrics(analysis_id="invalid-tokens")

    arguments = {
        "model": "gemini-2.5-flash",
        "duration_ms": 1,
        "prompt_tokens": 1,
        "output_tokens": 1,
        "thinking_tokens": 0,
        "total_tokens": 2,
    }

    arguments[field] = value

    with pytest.raises(error):
        analysis.record_success(**arguments)  # type: ignore[arg-type]

    snapshot = analysis.snapshot()

    assert snapshot["requests"]["total"] == 0
    assert snapshot["tokens"]["total"] == 0
    assert snapshot["token_cost_estimate"]["requests_estimated"] == 0


# ============================================================
# CONTEXTO INDIVIDUAL POR ANÁLISE
# ============================================================


def test_nested_analysis_context_restores_outer_context(
    metrics: LLMMetrics,
) -> None:
    with metrics.track_analysis("outer") as outer:
        metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=1,
            prompt_tokens=1,
            output_tokens=1,
        )

        with metrics.track_analysis("inner") as inner:
            metrics.record_success(
                model="gemini-2.5-flash",
                duration_ms=1,
                prompt_tokens=2,
                output_tokens=2,
            )

        metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=1,
            prompt_tokens=3,
            output_tokens=3,
        )

    assert inner.snapshot()["requests"]["total"] == 1
    assert outer.snapshot()["requests"]["total"] == 2
    assert outer.snapshot()["tokens"]["input"] == 4


def test_analysis_context_is_restored_after_exception(
    metrics: LLMMetrics,
) -> None:
    with metrics.track_analysis("outer") as outer:
        with pytest.raises(RuntimeError):
            with metrics.track_analysis("inner"):
                raise RuntimeError("failure")

        metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=1,
            prompt_tokens=1,
            output_tokens=1,
        )

    assert outer.snapshot()["requests"]["total"] == 1


def test_analysis_context_does_not_leak_after_exit(
    metrics: LLMMetrics,
) -> None:
    with metrics.track_analysis("scoped") as scoped:
        metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=1,
            prompt_tokens=1,
            output_tokens=1,
        )

    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=1,
        prompt_tokens=2,
        output_tokens=2,
    )

    assert scoped.snapshot()["requests"]["total"] == 1
    assert metrics.snapshot()["requests"]["total"] == 2


def test_analysis_metrics_direct_record_success_estimates_cost() -> None:
    analysis = AnalysisLLMMetrics(
        analysis_id="direct-analysis",
        provider="openai",
        model="gpt-4.1-mini",
    )

    analysis.record_success(
        model="gpt-4.1-mini",
        duration_ms=10,
        prompt_tokens=1_000,
        output_tokens=500,
    )

    assert analysis.snapshot()["token_cost_estimate"]["estimated_cost_usd"] == (
        pytest.approx(0.0012)
    )


# ============================================================
# TRACING
# ============================================================


def test_trace_filtering_and_retention() -> None:
    metrics = LLMMetrics(max_trace_events=2)

    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=1,
        prompt_tokens=1,
        output_tokens=1,
    )

    metrics.record_failure(
        model="gemini-2.5-flash",
        duration_ms=2,
    )

    metrics.record_retry(
        provider="openai",
        model="gpt-4.1-mini",
    )

    traces = metrics.get_recent_traces(limit=10)

    assert len(traces) == 2
    assert traces[0]["event_type"] == "llm_retry"

    gemini_traces = metrics.get_recent_traces(provider="gemini")

    assert len(gemini_traces) == 1
    assert gemini_traces[0]["status"] == "failure"


# ============================================================
# CONCORRÊNCIA ENTRE THREADS
# ============================================================


def test_concurrent_threads_preserve_global_and_analysis_metrics(
    metrics: LLMMetrics,
) -> None:
    worker_count = 8
    requests_per_worker = 100

    barrier = Barrier(worker_count)

    def record_batch(worker_index: int) -> dict:
        with metrics.track_analysis(f"thread-analysis-{worker_index}") as analysis:
            barrier.wait()

            for _ in range(requests_per_worker):
                metrics.record_retry(provider="openai")

                metrics.record_success(
                    model="gpt-4.1-mini",
                    duration_ms=1.0,
                    prompt_tokens=1,
                    output_tokens=2,
                    total_tokens=3,
                    provider="openai",
                )

            return analysis.snapshot()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        analysis_results = list(
            executor.map(
                record_batch,
                range(worker_count),
            )
        )

    expected_total = worker_count * requests_per_worker

    global_result = metrics.snapshot("openai")

    assert global_result["requests"]["total"] == expected_total
    assert global_result["requests"]["successful"] == expected_total
    assert global_result["requests"]["failed"] == 0
    assert global_result["requests"]["retries"] == expected_total

    assert global_result["tokens"]["input"] == expected_total
    assert global_result["tokens"]["output"] == expected_total * 2
    assert global_result["tokens"]["total"] == expected_total * 3

    assert global_result["latency"]["average_ms"] == 1.0

    assert len(analysis_results) == worker_count

    for index, result in enumerate(analysis_results):
        assert result["analysis_id"] == f"thread-analysis-{index}"
        assert result["provider"] == "openai"

        assert result["requests"]["total"] == requests_per_worker
        assert result["requests"]["successful"] == requests_per_worker
        assert result["requests"]["failed"] == 0
        assert result["requests"]["retries"] == requests_per_worker

        assert result["tokens"]["input"] == requests_per_worker
        assert result["tokens"]["output"] == requests_per_worker * 2
        assert result["tokens"]["total"] == requests_per_worker * 3

        assert result["latency"]["total_ms"] == float(requests_per_worker)


# ============================================================
# CONCORRÊNCIA ASSÍNCRONA
# ============================================================


def test_async_tasks_preserve_independent_analysis_contexts(
    metrics: LLMMetrics,
) -> None:
    async def run_analysis(worker_index: int) -> dict:
        input_tokens = worker_index + 1

        with metrics.track_analysis(f"async-analysis-{worker_index}") as analysis:
            await asyncio.sleep(0)

            metrics.record_success(
                model="gpt-4.1-mini",
                duration_ms=10.0,
                prompt_tokens=input_tokens,
                output_tokens=2,
                total_tokens=input_tokens + 2,
                provider="openai",
            )

            await asyncio.sleep(0)

            metrics.record_success(
                model="gpt-4.1-mini",
                duration_ms=20.0,
                prompt_tokens=input_tokens,
                output_tokens=2,
                total_tokens=input_tokens + 2,
                provider="openai",
            )

            return analysis.snapshot()

    async def run_all() -> list[dict]:
        return await asyncio.gather(*(run_analysis(index) for index in range(4)))

    results = asyncio.run(run_all())

    assert len(results) == 4

    for index, result in enumerate(results):
        input_tokens = index + 1

        assert result["analysis_id"] == f"async-analysis-{index}"
        assert result["provider"] == "openai"

        assert result["requests"]["total"] == 2
        assert result["requests"]["successful"] == 2

        assert result["tokens"]["input"] == input_tokens * 2
        assert result["tokens"]["output"] == 4
        assert result["tokens"]["total"] == (input_tokens + 2) * 2

        assert result["latency"]["total_ms"] == 30.0
        assert result["latency"]["average_ms"] == 15.0

    global_result = metrics.snapshot("openai")

    assert global_result["requests"]["total"] == 8
    assert global_result["requests"]["successful"] == 8
    assert global_result["tokens"]["output"] == 16
