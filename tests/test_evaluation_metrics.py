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

    # Resumo geral.
    overall = metrics["summary"]["analysis_mode_adherence"]

    assert overall == {
        "expected_cases": 3,
        "observed_cases": 2,
        "unobserved_cases": 1,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }

    # Categoria com dois modos observados.
    category = metrics["by_category"]["operational_overview"]["analysis_mode_adherence"]

    assert category == {
        "expected_cases": 2,
        "observed_cases": 2,
        "unobserved_cases": 0,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }

    # Categoria em que nenhum modo pôde ser observado.
    delivery = metrics["by_category"]["delivery"]["analysis_mode_adherence"]

    assert delivery == {
        "expected_cases": 1,
        "observed_cases": 0,
        "unobserved_cases": 1,
        "matches": 0,
        "mismatches": 0,
        "rate": None,
    }

    # Agrupamento por provedor.
    gemini = metrics["by_provider"]["gemini"]["analysis_mode_adherence"]

    assert gemini["expected_cases"] == 2
    assert gemini["rate"] == 50.0

    openai = metrics["by_provider"]["openai"]["analysis_mode_adherence"]

    assert openai["expected_cases"] == 1
    assert openai["unobserved_cases"] == 1
    assert openai["rate"] is None

    # Agrupamento por modelo.
    flash = metrics["by_model"]["gemini/gemini-2.5-flash"]["analysis_mode_adherence"]

    assert flash["matches"] == 1
    assert flash["mismatches"] == 1
    assert flash["rate"] == 50.0
