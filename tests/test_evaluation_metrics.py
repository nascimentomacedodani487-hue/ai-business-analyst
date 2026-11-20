from __future__ import annotations

from ai_analyst.evaluation.metrics import (
    calculate_evaluation_metrics,
)
from ai_analyst.evaluation.models import EvaluationCaseResult


def make_result(
    case_id,
    status="passed",
    score=1.0,
    duration_ms=None,
    cache_hit=None,
    category="volume",
    provider="gemini",
    model="gemini-2.5-flash",
    expected_analysis_mode=None,
    analysis_mode=None,
    quality_metrics=None,
    llm_metrics=None,
):
    metadata = {
        "category": category,
    }

    if expected_analysis_mode is not None:
        metadata["expected_analysis_mode"] = expected_analysis_mode

    if analysis_mode is not None:
        metadata["analysis_mode"] = analysis_mode

        if expected_analysis_mode is not None:
            metadata["analysis_mode_matches"] = analysis_mode == expected_analysis_mode

    if quality_metrics is not None:
        metadata.update(quality_metrics)

    if llm_metrics is not None:
        metadata["llm_metrics"] = llm_metrics

    return EvaluationCaseResult(
        case_id=case_id,
        question=f"Question {case_id}",
        status=status,
        score=score,
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        provider=provider,
        model=model,
        metadata=metadata,
    )


def make_llm_snapshot(
    provider="gemini",
    model="gemini-2.5-flash",
    requests_total=1,
    requests_successful=1,
    requests_failed=0,
    retries=0,
    input_tokens=100,
    output_tokens=50,
    thinking_tokens=0,
    total_tokens=150,
    tokens_unreported=0,
    total_latency_ms=100.0,
    average_latency_ms=100.0,
):
    """Cria um snapshot individual simulado de consumo de LLM."""

    return {
        "analysis_id": "analysis-test-001",
        "provider": provider,
        "model": model,
        "requests": {
            "total": requests_total,
            "successful": requests_successful,
            "failed": requests_failed,
            "retries": retries,
        },
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "thinking": thinking_tokens,
            "total": total_tokens,
            "unreported": tokens_unreported,
        },
        "latency": {
            "total_ms": total_latency_ms,
            "average_ms": average_latency_ms,
        },
    }


# ==========================================================
# MÉTRICAS BÁSICAS
# ==========================================================


def test_empty_results_return_zero_metrics():
    metrics = calculate_evaluation_metrics([])

    summary = metrics["summary"]

    assert summary["total_cases"] == 0
    assert summary["evaluated_cases"] == 0
    assert summary["passed_cases"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["average_score"] == 0.0
    assert summary["average_duration_ms"] is None
    assert summary["cache_hit_rate"] is None

    mode_metrics = summary["analysis_mode_adherence"]

    assert mode_metrics == {
        "expected_cases": 0,
        "observed_cases": 0,
        "unobserved_cases": 0,
        "matches": 0,
        "mismatches": 0,
        "rate": None,
    }

    assert summary["matched_rows"] == 0
    assert summary["compared_rows"] == 0
    assert summary["row_metric_cases"] == 0
    assert summary["row_accuracy"] is None

    assert summary["matched_cells"] == 0
    assert summary["compared_cells"] == 0
    assert summary["cell_metric_cases"] == 0
    assert summary["cell_accuracy"] is None

    assert summary["llm_usage"]["observed_cases"] == 0
    assert summary["llm_usage"]["requests"]["total"] == 0
    assert summary["llm_usage"]["tokens"]["total"] == 0
    assert summary["llm_usage"]["cost_estimate_usd"] is None


def test_summary_counts_statuses_correctly():
    results = [
        make_result("1", status="passed"),
        make_result("2", status="passed"),
        make_result("3", status="failed", score=0.5),
        make_result("4", status="error"),
        make_result("5", status="skipped"),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["total_cases"] == 5
    assert summary["evaluated_cases"] == 4
    assert summary["passed_cases"] == 2
    assert summary["failed_cases"] == 1
    assert summary["error_cases"] == 1
    assert summary["skipped_cases"] == 1


def test_pass_rate_includes_errors_but_excludes_skipped():
    results = [
        make_result("1", status="passed"),
        make_result("2", status="failed"),
        make_result("3", status="error"),
        make_result("4", status="skipped"),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["pass_rate"] == 33.33


def test_average_score_excludes_errors_and_skipped():
    results = [
        make_result("1", status="passed", score=1.0),
        make_result("2", status="failed", score=0.5),
        make_result("3", status="error", score=0.0),
        make_result("4", status="skipped", score=0.0),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["scored_cases"] == 2
    assert summary["average_score"] == 0.75


def test_average_duration_uses_available_measurements():
    results = [
        make_result("1", duration_ms=100),
        make_result("2", duration_ms=300),
        make_result("3", duration_ms=None),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["average_duration_ms"] == 200.0


def test_cache_rate_ignores_unknown_cache_values():
    results = [
        make_result("1", cache_hit=True),
        make_result("2", cache_hit=True),
        make_result("3", cache_hit=False),
        make_result("4", cache_hit=None),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["cache_observations"] == 3
    assert summary["cache_hits"] == 2
    assert summary["cache_misses"] == 1
    assert summary["cache_hit_rate"] == 66.67


# ==========================================================
# AGRUPAMENTO
# ==========================================================


def test_metrics_are_grouped_by_category():
    results = [
        make_result("1", category="volume"),
        make_result("2", category="volume"),
        make_result(
            "3",
            category="financial",
            status="failed",
            score=0.5,
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    assert set(metrics["by_category"]) == {
        "volume",
        "financial",
    }

    assert metrics["by_category"]["volume"]["passed_cases"] == 2

    assert metrics["by_category"]["financial"]["failed_cases"] == 1


def test_metrics_are_grouped_by_provider():
    results = [
        make_result("1", provider="gemini"),
        make_result("2", provider="gemini"),
        make_result("3", provider="openai"),
    ]

    metrics = calculate_evaluation_metrics(results)

    assert set(metrics["by_provider"]) == {
        "gemini",
        "openai",
    }

    assert metrics["by_provider"]["gemini"]["total_cases"] == 2

    assert metrics["by_provider"]["openai"]["total_cases"] == 1


def test_metrics_distinguish_models_of_same_provider():
    results = [
        make_result(
            "1",
            provider="gemini",
            model="gemini-2.5-flash",
        ),
        make_result(
            "2",
            provider="gemini",
            model="gemini-2.5-pro",
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    assert set(metrics["by_model"]) == {
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
    }


def test_missing_provider_and_model_use_unknown_group():
    results = [
        make_result(
            "1",
            provider=None,
            model=None,
        )
    ]

    metrics = calculate_evaluation_metrics(results)

    assert "unknown" in metrics["by_provider"]
    assert "unknown/unknown" in metrics["by_model"]


def test_accepts_generators_as_input():
    results = (make_result(str(index)) for index in range(3))

    metrics = calculate_evaluation_metrics(results)

    assert metrics["summary"]["total_cases"] == 3


# ==========================================================
# ADERÊNCIA AO MODO DE ANÁLISE
# ==========================================================


def test_analysis_mode_adherence_counts_matching_modes():
    results = [
        make_result(
            "1",
            expected_analysis_mode="multi_query",
            analysis_mode="multi_query",
        ),
        make_result(
            "2",
            expected_analysis_mode="single_query",
            analysis_mode="single_query",
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    adherence = summary["analysis_mode_adherence"]

    assert adherence == {
        "expected_cases": 2,
        "observed_cases": 2,
        "unobserved_cases": 0,
        "matches": 2,
        "mismatches": 0,
        "rate": 100.0,
    }


def test_analysis_mode_adherence_counts_mismatches():
    results = [
        make_result(
            "1",
            expected_analysis_mode="multi_query",
            analysis_mode="single_query",
            status="failed",
        ),
        make_result(
            "2",
            expected_analysis_mode="single_query",
            analysis_mode="multi_query",
            status="failed",
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    adherence = summary["analysis_mode_adherence"]

    assert adherence == {
        "expected_cases": 2,
        "observed_cases": 2,
        "unobserved_cases": 0,
        "matches": 0,
        "mismatches": 2,
        "rate": 0.0,
    }


def test_analysis_mode_adherence_excludes_unobserved_cases_from_rate():
    results = [
        make_result(
            "1",
            expected_analysis_mode="multi_query",
            analysis_mode="multi_query",
        ),
        make_result(
            "2",
            expected_analysis_mode="multi_query",
            analysis_mode="single_query",
            status="failed",
        ),
        make_result(
            "3",
            expected_analysis_mode="single_query",
            analysis_mode=None,
            status="error",
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    adherence = summary["analysis_mode_adherence"]

    assert adherence == {
        "expected_cases": 3,
        "observed_cases": 2,
        "unobserved_cases": 1,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }


def test_analysis_mode_adherence_includes_observed_modes_even_when_status_is_error():
    """Um erro posterior à análise não apaga o modo observado."""

    results = [
        make_result(
            "1",
            expected_analysis_mode="multi_query",
            analysis_mode="multi_query",
            status="error",
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    adherence = summary["analysis_mode_adherence"]

    assert adherence["expected_cases"] == 1
    assert adherence["observed_cases"] == 1
    assert adherence["unobserved_cases"] == 0
    assert adherence["matches"] == 1
    assert adherence["mismatches"] == 0
    assert adherence["rate"] == 100.0


def test_analysis_mode_adherence_is_none_without_expectations():
    results = [
        make_result(
            "1",
            analysis_mode="single_query",
        ),
        make_result(
            "2",
            analysis_mode="multi_query",
        ),
        make_result("3"),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    adherence = summary["analysis_mode_adherence"]

    assert adherence == {
        "expected_cases": 0,
        "observed_cases": 0,
        "unobserved_cases": 0,
        "matches": 0,
        "mismatches": 0,
        "rate": None,
    }


def test_analysis_mode_adherence_is_grouped_by_category_provider_and_model():
    results = [
        make_result(
            "1",
            category="operational_overview",
            provider="gemini",
            model="gemini-2.5-flash",
            expected_analysis_mode="multi_query",
            analysis_mode="multi_query",
        ),
        make_result(
            "2",
            category="operational_overview",
            provider="gemini",
            model="gemini-2.5-flash",
            expected_analysis_mode="multi_query",
            analysis_mode="single_query",
            status="failed",
        ),
        make_result(
            "3",
            category="delivery",
            provider="openai",
            model="gpt-4.1-mini",
            expected_analysis_mode="single_query",
            status="error",
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    overall = metrics["summary"]["analysis_mode_adherence"]

    assert overall == {
        "expected_cases": 3,
        "observed_cases": 2,
        "unobserved_cases": 1,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }

    category = metrics["by_category"]["operational_overview"]["analysis_mode_adherence"]

    assert category == {
        "expected_cases": 2,
        "observed_cases": 2,
        "unobserved_cases": 0,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }

    delivery = metrics["by_category"]["delivery"]["analysis_mode_adherence"]

    assert delivery == {
        "expected_cases": 1,
        "observed_cases": 0,
        "unobserved_cases": 1,
        "matches": 0,
        "mismatches": 0,
        "rate": None,
    }

    gemini = metrics["by_provider"]["gemini"]["analysis_mode_adherence"]

    assert gemini["expected_cases"] == 2
    assert gemini["rate"] == 50.0

    openai = metrics["by_provider"]["openai"]["analysis_mode_adherence"]

    assert openai["expected_cases"] == 1
    assert openai["unobserved_cases"] == 1
    assert openai["rate"] is None

    flash = metrics["by_model"]["gemini/gemini-2.5-flash"]["analysis_mode_adherence"]

    assert flash["matches"] == 1
    assert flash["mismatches"] == 1
    assert flash["rate"] == 50.0


# ==========================================================
# MÉTRICAS DE QUALIDADE DOS RESULTADOS SQL
# ==========================================================


def test_result_quality_metrics_are_aggregated():
    """Verifica a consolidação dos totais de linhas e células."""

    results = [
        make_result(
            "1",
            quality_metrics={
                "matched_rows": 1,
                "compared_rows": 1,
                "matched_cells": 2,
                "compared_cells": 2,
            },
        ),
        make_result(
            "2",
            status="failed",
            score=0.5,
            quality_metrics={
                "matched_rows": 0,
                "compared_rows": 3,
                "matched_cells": 3,
                "compared_cells": 6,
            },
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["matched_rows"] == 1
    assert summary["compared_rows"] == 4
    assert summary["row_metric_cases"] == 2
    assert summary["row_accuracy"] == 25.0

    assert summary["matched_cells"] == 5
    assert summary["compared_cells"] == 8
    assert summary["cell_metric_cases"] == 2
    assert summary["cell_accuracy"] == 62.5


def test_result_quality_metrics_are_weighted_by_comparison_volume():
    """Percentuais agregados não devem ser médias simples dos cenários."""

    results = [
        make_result(
            "small_case",
            quality_metrics={
                "matched_rows": 1,
                "compared_rows": 1,
                "matched_cells": 2,
                "compared_cells": 2,
            },
        ),
        make_result(
            "large_case",
            quality_metrics={
                "matched_rows": 0,
                "compared_rows": 9,
                "matched_cells": 0,
                "compared_cells": 18,
            },
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["row_accuracy"] == 10.0
    assert summary["cell_accuracy"] == 10.0


def test_result_quality_metrics_are_grouped_by_category_provider_and_model():
    results = [
        make_result(
            "1",
            category="volume",
            provider="gemini",
            model="gemini-2.5-flash",
            quality_metrics={
                "matched_rows": 1,
                "compared_rows": 1,
                "matched_cells": 2,
                "compared_cells": 2,
            },
        ),
        make_result(
            "2",
            category="volume",
            provider="gemini",
            model="gemini-2.5-flash",
            quality_metrics={
                "matched_rows": 0,
                "compared_rows": 3,
                "matched_cells": 3,
                "compared_cells": 6,
            },
        ),
        make_result(
            "3",
            category="financial",
            provider="openai",
            model="gpt-4.1-mini",
            quality_metrics={
                "matched_rows": 1,
                "compared_rows": 1,
                "matched_cells": 3,
                "compared_cells": 3,
            },
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    overall = metrics["summary"]

    assert overall["row_accuracy"] == 40.0
    assert overall["cell_accuracy"] == 72.73

    volume = metrics["by_category"]["volume"]

    assert volume["matched_rows"] == 1
    assert volume["compared_rows"] == 4
    assert volume["row_accuracy"] == 25.0
    assert volume["cell_accuracy"] == 62.5

    financial = metrics["by_category"]["financial"]

    assert financial["row_accuracy"] == 100.0
    assert financial["cell_accuracy"] == 100.0

    gemini = metrics["by_provider"]["gemini"]

    assert gemini["row_accuracy"] == 25.0
    assert gemini["cell_accuracy"] == 62.5

    openai = metrics["by_provider"]["openai"]

    assert openai["row_accuracy"] == 100.0
    assert openai["cell_accuracy"] == 100.0

    flash = metrics["by_model"]["gemini/gemini-2.5-flash"]

    assert flash["row_accuracy"] == 25.0
    assert flash["cell_accuracy"] == 62.5

    gpt = metrics["by_model"]["openai/gpt-4.1-mini"]

    assert gpt["row_accuracy"] == 100.0
    assert gpt["cell_accuracy"] == 100.0


def test_result_quality_metrics_ignore_legacy_results_without_metadata():
    """Relatórios antigos continuam compatíveis."""

    results = [
        make_result(
            "1",
            quality_metrics={
                "matched_rows": 2,
                "compared_rows": 4,
                "matched_cells": 4,
                "compared_cells": 8,
            },
        ),
        make_result("2"),
        make_result("3"),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["matched_rows"] == 2
    assert summary["compared_rows"] == 4
    assert summary["row_metric_cases"] == 1
    assert summary["row_accuracy"] == 50.0

    assert summary["matched_cells"] == 4
    assert summary["compared_cells"] == 8
    assert summary["cell_metric_cases"] == 1
    assert summary["cell_accuracy"] == 50.0


def test_result_quality_metrics_ignore_invalid_counters():
    results = [
        make_result(
            "valid",
            quality_metrics={
                "matched_rows": 2,
                "compared_rows": 4,
                "matched_cells": 6,
                "compared_cells": 10,
            },
        ),
        make_result(
            "invalid",
            quality_metrics={
                "matched_rows": 5,
                "compared_rows": 2,
                "matched_cells": -1,
                "compared_cells": 3,
            },
        ),
        make_result(
            "boolean",
            quality_metrics={
                "matched_rows": True,
                "compared_rows": 2,
                "matched_cells": 1,
                "compared_cells": False,
            },
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["matched_rows"] == 2
    assert summary["compared_rows"] == 4
    assert summary["row_metric_cases"] == 1
    assert summary["row_accuracy"] == 50.0

    assert summary["matched_cells"] == 6
    assert summary["compared_cells"] == 10
    assert summary["cell_metric_cases"] == 1
    assert summary["cell_accuracy"] == 60.0


def test_result_quality_metrics_return_none_without_comparable_rows_or_cells():
    results = [
        make_result(
            "1",
            quality_metrics={
                "matched_rows": 0,
                "compared_rows": 0,
                "matched_cells": 0,
                "compared_cells": 0,
            },
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["matched_rows"] == 0
    assert summary["compared_rows"] == 0
    assert summary["row_metric_cases"] == 1
    assert summary["row_accuracy"] is None

    assert summary["matched_cells"] == 0
    assert summary["compared_cells"] == 0
    assert summary["cell_metric_cases"] == 1
    assert summary["cell_accuracy"] is None


def test_result_quality_metrics_validate_counters_independently():
    """Uma métrica válida não deve ser descartada por outra inválida."""

    results = [
        make_result(
            "1",
            quality_metrics={
                "matched_rows": 1,
                "compared_rows": 2,
                "matched_cells": 5,
                "compared_cells": 3,
            },
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]

    assert summary["row_metric_cases"] == 1
    assert summary["row_accuracy"] == 50.0

    assert summary["cell_metric_cases"] == 0
    assert summary["cell_accuracy"] is None


# ==========================================================
# MÉTRICAS DE CONSUMO DE LLM
# ==========================================================


def test_llm_usage_aggregates_requests_tokens_retries_and_latency():
    """Consolida o consumo individual de vários cenários."""

    results = [
        make_result(
            "1",
            llm_metrics=make_llm_snapshot(
                requests_total=2,
                requests_successful=2,
                requests_failed=0,
                retries=1,
                input_tokens=100,
                output_tokens=50,
                thinking_tokens=10,
                total_tokens=160,
                total_latency_ms=200.0,
                average_latency_ms=100.0,
            ),
        ),
        make_result(
            "2",
            llm_metrics=make_llm_snapshot(
                requests_total=1,
                requests_successful=0,
                requests_failed=1,
                retries=0,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                total_tokens=0,
                tokens_unreported=1,
                total_latency_ms=100.0,
                average_latency_ms=100.0,
            ),
        ),
    ]

    summary = calculate_evaluation_metrics(results)["summary"]
    usage = summary["llm_usage"]

    assert usage["observed_cases"] == 2
    assert usage["cases_with_requests"] == 2
    assert usage["cases_without_requests"] == 0

    assert usage["providers"] == ["gemini"]

    assert usage["requests"] == {
        "total": 3,
        "successful": 2,
        "failed": 1,
        "retries": 1,
        "success_rate": 66.67,
        "failure_rate": 33.33,
    }

    assert usage["tokens"] == {
        "input": 100,
        "output": 50,
        "thinking": 10,
        "total": 160,
        "unreported": 1,
    }

    assert usage["latency"] == {
        "total_ms": 300.0,
        "average_request_ms": 100.0,
        "average_case_ms": 150.0,
        "measured_cases": 2,
    }

    assert usage["pricing_tier"] == "free"
    assert usage["cost_estimate_usd"] == 0.0

    assert usage["pricing_by_provider"]["gemini"]["cost_estimate_usd"] == 0.0


def test_llm_usage_calculates_request_rates_correctly():
    results = [
        make_result(
            "1",
            llm_metrics=make_llm_snapshot(
                requests_total=10,
                requests_successful=7,
                requests_failed=3,
            ),
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["requests"]["success_rate"] == 70.0
    assert usage["requests"]["failure_rate"] == 30.0


def test_llm_usage_does_not_invent_cost_for_usage_based_providers():
    """OpenAI e Anthropic não recebem estimativas tarifárias artificiais."""

    results = [
        make_result(
            "1",
            provider="openai",
            model="gpt-4.1-mini",
            llm_metrics=make_llm_snapshot(
                provider="openai",
                model="gpt-4.1-mini",
            ),
        ),
        make_result(
            "2",
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            llm_metrics=make_llm_snapshot(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
            ),
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["pricing_tier"] == "usage_based"
    assert usage["cost_estimate_usd"] is None

    assert usage["pricing_by_provider"]["openai"]["cost_estimate_usd"] is None
    assert usage["pricing_by_provider"]["anthropic"]["cost_estimate_usd"] is None


def test_llm_usage_marks_pricing_as_mixed_when_providers_differ():
    results = [
        make_result(
            "1",
            provider="gemini",
            llm_metrics=make_llm_snapshot(
                provider="gemini",
            ),
        ),
        make_result(
            "2",
            provider="openai",
            model="gpt-4.1-mini",
            llm_metrics=make_llm_snapshot(
                provider="openai",
                model="gpt-4.1-mini",
            ),
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["pricing_tier"] == "mixed"
    assert usage["cost_estimate_usd"] is None

    assert usage["pricing_by_provider"]["gemini"]["pricing_tier"] == "free"
    assert usage["pricing_by_provider"]["openai"]["pricing_tier"] == "usage_based"


def test_llm_usage_is_grouped_by_category_provider_and_model():
    results = [
        make_result(
            "1",
            category="volume",
            provider="gemini",
            model="gemini-2.5-flash",
            llm_metrics=make_llm_snapshot(
                provider="gemini",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                total_latency_ms=100.0,
            ),
        ),
        make_result(
            "2",
            category="volume",
            provider="gemini",
            model="gemini-2.5-flash",
            llm_metrics=make_llm_snapshot(
                provider="gemini",
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
                total_latency_ms=200.0,
            ),
        ),
        make_result(
            "3",
            category="financial",
            provider="openai",
            model="gpt-4.1-mini",
            llm_metrics=make_llm_snapshot(
                provider="openai",
                model="gpt-4.1-mini",
                input_tokens=300,
                output_tokens=150,
                total_tokens=450,
                total_latency_ms=300.0,
            ),
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    overall = metrics["summary"]["llm_usage"]

    assert overall["observed_cases"] == 3
    assert overall["tokens"]["input"] == 600
    assert overall["tokens"]["output"] == 300
    assert overall["tokens"]["total"] == 900
    assert overall["latency"]["total_ms"] == 600.0

    volume = metrics["by_category"]["volume"]["llm_usage"]

    assert volume["observed_cases"] == 2
    assert volume["tokens"]["input"] == 300
    assert volume["tokens"]["total"] == 450
    assert volume["latency"]["total_ms"] == 300.0

    financial = metrics["by_category"]["financial"]["llm_usage"]

    assert financial["observed_cases"] == 1
    assert financial["tokens"]["total"] == 450

    gemini = metrics["by_provider"]["gemini"]["llm_usage"]

    assert gemini["observed_cases"] == 2
    assert gemini["tokens"]["total"] == 450

    openai = metrics["by_provider"]["openai"]["llm_usage"]

    assert openai["observed_cases"] == 1
    assert openai["tokens"]["total"] == 450

    flash = metrics["by_model"]["gemini/gemini-2.5-flash"]["llm_usage"]

    assert flash["observed_cases"] == 2
    assert flash["tokens"]["total"] == 450

    gpt = metrics["by_model"]["openai/gpt-4.1-mini"]["llm_usage"]

    assert gpt["observed_cases"] == 1
    assert gpt["tokens"]["total"] == 450


def test_llm_usage_ignores_legacy_results_without_snapshots():
    """Relatórios anteriores à integração continuam compatíveis."""

    results = [
        make_result("1"),
        make_result(
            "2",
            llm_metrics=make_llm_snapshot(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        ),
        make_result("3"),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["observed_cases"] == 1
    assert usage["cases_with_requests"] == 1
    assert usage["cases_without_requests"] == 0

    assert usage["requests"]["total"] == 1
    assert usage["tokens"]["total"] == 150


def test_llm_usage_ignores_invalid_snapshots():
    """Snapshots incompletos não devem contaminar a consolidação."""

    results = [
        make_result("1"),
        make_result(
            "2",
            llm_metrics={
                "provider": "gemini",
                "requests": {
                    "total": 1,
                },
                "tokens": {
                    "total": 100,
                },
            },
        ),
        make_result(
            "3",
            llm_metrics="invalid-snapshot",
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["observed_cases"] == 0
    assert usage["cases_with_requests"] == 0
    assert usage["cases_without_requests"] == 0

    assert usage["requests"]["total"] == 0
    assert usage["tokens"]["total"] == 0

    assert usage["pricing_tier"] == "unknown"
    assert usage["cost_estimate_usd"] is None


def test_llm_usage_counts_cases_without_requests():
    """Um cenário pode possuir snapshot válido sem chamadas registradas."""

    results = [
        make_result(
            "1",
            llm_metrics=make_llm_snapshot(
                requests_total=0,
                requests_successful=0,
                requests_failed=0,
                retries=0,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                total_tokens=0,
                tokens_unreported=0,
                total_latency_ms=0.0,
                average_latency_ms=0.0,
            ),
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["observed_cases"] == 1
    assert usage["cases_with_requests"] == 0
    assert usage["cases_without_requests"] == 1

    assert usage["requests"]["total"] == 0
    assert usage["requests"]["success_rate"] is None
    assert usage["requests"]["failure_rate"] is None

    assert usage["latency"]["average_request_ms"] is None
    assert usage["latency"]["average_case_ms"] == 0.0

    assert usage["pricing_tier"] == "free"


def test_llm_usage_uses_snapshot_provider_when_result_provider_differs():
    """O provedor registrado no consumo individual tem prioridade."""

    results = [
        make_result(
            "1",
            provider="gemini",
            llm_metrics=make_llm_snapshot(
                provider="openai",
                model="gpt-4.1-mini",
                input_tokens=200,
                output_tokens=100,
                total_tokens=300,
            ),
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["providers"] == ["openai"]
    assert usage["pricing_tier"] == "usage_based"
    assert usage["cost_estimate_usd"] is None

    assert usage["pricing_by_provider"]["openai"]["cost_estimate_usd"] is None


def test_llm_usage_ignores_invalid_numeric_values():
    """Contadores e latências inválidos não entram na agregação."""

    results = [
        make_result(
            "1",
            llm_metrics={
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "requests": {
                    "total": True,
                    "successful": -1,
                    "failed": 1.5,
                    "retries": 2,
                },
                "tokens": {
                    "input": True,
                    "output": -10,
                    "thinking": 15,
                    "total": 50,
                    "unreported": 0,
                },
                "latency": {
                    "total_ms": float("inf"),
                    "average_ms": 100.0,
                },
            },
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["observed_cases"] == 1

    assert usage["requests"]["total"] == 0
    assert usage["requests"]["successful"] == 0
    assert usage["requests"]["failed"] == 0
    assert usage["requests"]["retries"] == 2

    assert usage["tokens"]["input"] == 0
    assert usage["tokens"]["output"] == 0
    assert usage["tokens"]["thinking"] == 15
    assert usage["tokens"]["total"] == 50

    assert usage["latency"]["total_ms"] == 0.0
    assert usage["latency"]["measured_cases"] == 0


def test_llm_usage_does_not_multiply_provider_pricing_metadata_by_case_count():
    """Metadados de cobrança não são tratados como preço por chamada."""

    results = [
        make_result(
            "1",
            llm_metrics=make_llm_snapshot(provider="gemini"),
        ),
        make_result(
            "2",
            llm_metrics=make_llm_snapshot(provider="gemini"),
        ),
        make_result(
            "3",
            llm_metrics=make_llm_snapshot(provider="gemini"),
        ),
    ]

    usage = calculate_evaluation_metrics(results)["summary"]["llm_usage"]

    assert usage["observed_cases"] == 3
    assert usage["requests"]["total"] == 3

    assert usage["pricing_by_provider"]["gemini"]["cost_estimate_usd"] == 0.0
    assert usage["cost_estimate_usd"] == 0.0
