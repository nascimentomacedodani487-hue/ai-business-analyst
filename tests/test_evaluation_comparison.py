"""Testes do comparador histórico de benchmarks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from ai_analyst.evaluation.comparison import BenchmarkComparator

# ==========================================================
# FÁBRICAS DE DADOS
# ==========================================================


def make_provider_metrics(
    *,
    pass_rate: float = 100.0,
    average_score: float = 1.0,
    average_duration_ms: float = 100.0,
    row_accuracy: float = 100.0,
    cell_accuracy: float = 100.0,
    error_cases: int = 0,
    failed_cases: int = 0,
    skipped_cases: int = 0,
    request_total: int = 10,
    request_failed: int = 0,
    retries: int = 0,
    total_tokens: int = 1000,
    average_request_ms: float = 50.0,
) -> dict[str, Any]:
    """Cria métricas consolidadas fictícias de um provedor."""

    return {
        "total_cases": 2,
        "evaluated_cases": 2,
        "passed_cases": 2,
        "failed_cases": failed_cases,
        "error_cases": error_cases,
        "skipped_cases": skipped_cases,
        "pass_rate": pass_rate,
        "average_score": average_score,
        "average_duration_ms": average_duration_ms,
        "row_accuracy": row_accuracy,
        "cell_accuracy": cell_accuracy,
        "llm_usage": {
            "observed_cases": 2,
            "cases_with_requests": 2,
            "cases_without_requests": 0,
            "providers": ["gemini"],
            "requests": {
                "total": request_total,
                "successful": request_total - request_failed,
                "failed": request_failed,
                "retries": retries,
                "success_rate": (
                    round((request_total - request_failed) / request_total * 100, 2)
                    if request_total
                    else None
                ),
                "failure_rate": (
                    round(request_failed / request_total * 100, 2)
                    if request_total
                    else None
                ),
            },
            "tokens": {
                "input": total_tokens // 2,
                "output": total_tokens // 2,
                "thinking": 0,
                "total": total_tokens,
                "unreported": 0,
            },
            "latency": {
                "total_ms": average_request_ms * request_total,
                "average_request_ms": average_request_ms,
                "average_case_ms": average_duration_ms,
                "measured_cases": 2,
            },
            "pricing_tier": "free",
            "cost_estimate_usd": 0.0,
            "pricing_by_provider": {
                "gemini": {
                    "pricing_tier": "free",
                    "cost_estimate_usd": 0.0,
                },
            },
        },
    }


def make_benchmark_report(
    *,
    benchmark_id: str = "benchmark-001",
    dataset_sha256: str = "a" * 64,
    case_ids: tuple[str, ...] = ("case-1", "case-2"),
    providers: tuple[str, ...] = ("gemini", "openai"),
    models: dict[str, str] | None = None,
    metrics: dict[str, dict[str, Any]] | None = None,
    execution_statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Monta um benchmark consolidado fictício."""

    models = models or {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4.1-mini",
    }

    metrics = metrics or {
        "gemini": make_provider_metrics(),
        "openai": make_provider_metrics(),
    }

    execution_statuses = execution_statuses or {
        provider: "completed" for provider in providers
    }

    provider_runs = {}
    by_provider = {}

    completed_count = 0
    failed_count = 0

    for provider in providers:
        execution_status = execution_statuses[provider]
        model = models.get(provider)

        if execution_status == "completed":
            completed_count += 1

            individual_report = {
                "run_id": f"{benchmark_id}-{provider}",
                "provider": provider,
                "model": model,
                "summary": {
                    "total_cases": len(case_ids),
                    "evaluated_cases": len(case_ids),
                    "passed_cases": len(case_ids),
                    "failed_cases": 0,
                    "error_cases": 0,
                    "skipped_cases": 0,
                    "pass_rate": 100.0,
                    "duration_ms": 200.0,
                },
                "results": [],
            }
        else:
            failed_count += 1
            individual_report = None

        provider_runs[provider] = {
            "execution_status": execution_status,
            "report": individual_report,
            "error": None if execution_status == "completed" else "Falha simulada",
            "error_type": None if execution_status == "completed" else "RuntimeError",
        }

        by_provider[provider] = {
            "execution_status": execution_status,
            "model": model,
            "run_id": f"{benchmark_id}-{provider}",
            "metrics": metrics.get(provider)
            if execution_status == "completed"
            else None,
        }

    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": "2026-11-12T10:00:00+00:00",
        "dataset": {
            "version": 1,
            "source": None,
            "case_count": len(case_ids),
            "case_ids": list(case_ids),
            "sha256": dataset_sha256,
        },
        "configuration": {
            "providers": list(providers),
            "provider_count": len(providers),
            "scenario_count": len(case_ids),
            "execution_strategy": "sequential",
            "same_dataset_for_all_providers": True,
        },
        "summary": {
            "provider_count": len(providers),
            "completed_providers": completed_count,
            "failed_providers": failed_count,
            "all_providers_completed": failed_count == 0,
        },
        "provider_runs": provider_runs,
        "comparison": {
            "by_provider": by_provider,
            "by_category": {},
            "by_case": {},
        },
    }


@pytest.fixture
def baseline() -> dict[str, Any]:
    """Benchmark de referência."""

    return make_benchmark_report(
        benchmark_id="baseline-001",
    )


@pytest.fixture
def candidate(baseline: dict[str, Any]) -> dict[str, Any]:
    """Benchmark candidato inicialmente idêntico à referência."""

    report = deepcopy(baseline)
    report["benchmark_id"] = "candidate-001"
    report["created_at"] = "2026-11-12T11:00:00+00:00"

    return report


@pytest.fixture
def comparator() -> BenchmarkComparator:
    return BenchmarkComparator()


# ==========================================================
# COMPARAÇÃO BÁSICA
# ==========================================================


def test_comparison_returns_expected_structure(
    comparator,
    baseline,
    candidate,
):
    result = comparator.compare(baseline, candidate)

    assert result["schema_version"] == 1

    assert result["baseline"]["benchmark_id"] == "baseline-001"

    assert result["candidate"]["benchmark_id"] == "candidate-001"

    assert result["compatibility"]["comparable"] is True

    assert result["summary"]["status"] == "stable"

    assert set(result["providers"]) == {
        "gemini",
        "openai",
    }


def test_comparison_preserves_original_reports(
    comparator,
    baseline,
    candidate,
):
    original_baseline = deepcopy(baseline)
    original_candidate = deepcopy(candidate)

    comparator.compare(baseline, candidate)

    assert baseline == original_baseline
    assert candidate == original_candidate


def test_identical_benchmarks_are_classified_as_stable(
    comparator,
    baseline,
    candidate,
):
    result = comparator.compare(baseline, candidate)

    assert result["summary"]["status"] == "stable"

    assert result["summary"]["regressions"] == 0

    assert result["summary"]["improvements"] == 0


# ==========================================================
# COMPATIBILIDADE DOS DATASETS
# ==========================================================


def test_comparison_rejects_different_dataset_fingerprints(
    comparator,
    baseline,
    candidate,
):
    candidate["dataset"]["sha256"] = "b" * 64

    result = comparator.compare(baseline, candidate)

    assert result["compatibility"]["comparable"] is False

    assert result["summary"]["status"] == "incompatible"

    assert result["providers"] == {}

    assert result["compatibility"]["reasons"]


def test_comparison_rejects_different_case_identifiers(
    comparator,
    baseline,
    candidate,
):
    candidate["dataset"]["case_ids"] = [
        "case-1",
        "case-3",
    ]

    result = comparator.compare(baseline, candidate)

    assert result["compatibility"]["comparable"] is False

    assert result["summary"]["status"] == "incompatible"


def test_comparison_accepts_case_identifiers_in_different_order(
    comparator,
    baseline,
    candidate,
):
    candidate["dataset"]["case_ids"] = [
        "case-2",
        "case-1",
    ]

    result = comparator.compare(baseline, candidate)

    assert result["compatibility"]["comparable"] is True


def test_comparison_rejects_different_dataset_case_counts(
    comparator,
    baseline,
    candidate,
):
    candidate["dataset"]["case_count"] = 3

    result = comparator.compare(baseline, candidate)

    assert result["compatibility"]["comparable"] is False


# ==========================================================
# COMPATIBILIDADE DE PROVEDOR E MODELO
# ==========================================================


def test_comparison_compares_only_matching_providers(
    comparator,
    baseline,
    candidate,
):
    candidate["configuration"]["providers"] = ["gemini", "anthropic"]

    candidate["configuration"]["provider_count"] = 2

    candidate["provider_runs"].pop("openai")

    candidate["comparison"]["by_provider"].pop("openai")

    candidate["provider_runs"]["anthropic"] = {
        "execution_status": "completed",
        "report": {
            "model": "claude-haiku-4-5-20251001",
            "run_id": "candidate-anthropic",
        },
    }

    candidate["comparison"]["by_provider"]["anthropic"] = {
        "execution_status": "completed",
        "model": "claude-haiku-4-5-20251001",
        "metrics": make_provider_metrics(),
    }

    result = comparator.compare(baseline, candidate)

    assert result["providers"]["gemini"]["status"] == "compared"

    assert result["providers"]["openai"]["status"] == "baseline_only"

    assert result["providers"]["anthropic"]["status"] == "candidate_only"


def test_comparison_does_not_compare_different_models(
    comparator,
    baseline,
    candidate,
):
    candidate["provider_runs"]["gemini"]["report"]["model"] = "gemini-new-model"

    candidate["comparison"]["by_provider"]["gemini"]["model"] = "gemini-new-model"

    result = comparator.compare(baseline, candidate)

    provider_result = result["providers"]["gemini"]

    assert provider_result["status"] == "model_changed"

    assert provider_result["metrics"] == {}

    assert provider_result["baseline_model"] == "gemini-2.5-flash"

    assert provider_result["candidate_model"] == "gemini-new-model"


def test_comparison_does_not_compare_unknown_models(
    comparator,
    baseline,
    candidate,
):
    baseline["provider_runs"]["gemini"]["report"]["model"] = None

    baseline["comparison"]["by_provider"]["gemini"]["model"] = None

    candidate["provider_runs"]["gemini"]["report"]["model"] = None

    candidate["comparison"]["by_provider"]["gemini"]["model"] = None

    result = comparator.compare(baseline, candidate)

    assert result["providers"]["gemini"]["status"] == "model_unknown"

    assert result["providers"]["gemini"]["metrics"] == {}


# ==========================================================
# DIREÇÃO DAS MÉTRICAS
# ==========================================================


@pytest.mark.parametrize(
    (
        "metric_name",
        "baseline_value",
        "candidate_value",
        "expected_status",
    ),
    [
        ("pass_rate", 100.0, 80.0, "regression"),
        ("pass_rate", 80.0, 100.0, "improvement"),
        ("average_score", 1.0, 0.90, "regression"),
        ("average_score", 0.90, 1.0, "improvement"),
        ("row_accuracy", 100.0, 90.0, "regression"),
        ("row_accuracy", 90.0, 100.0, "improvement"),
        ("cell_accuracy", 100.0, 90.0, "regression"),
        ("cell_accuracy", 90.0, 100.0, "improvement"),
        ("average_duration_ms", 100.0, 200.0, "regression"),
        ("average_duration_ms", 200.0, 100.0, "improvement"),
        ("error_cases", 0, 2, "regression"),
        ("error_cases", 2, 0, "improvement"),
        ("failed_cases", 0, 1, "regression"),
        ("failed_cases", 1, 0, "improvement"),
        ("skipped_cases", 0, 1, "regression"),
        ("skipped_cases", 1, 0, "improvement"),
    ],
)
def test_comparison_classifies_metric_direction_correctly(
    comparator,
    baseline,
    candidate,
    metric_name,
    baseline_value,
    candidate_value,
    expected_status,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics[metric_name] = baseline_value

    candidate_metrics[metric_name] = candidate_value

    result = comparator.compare(baseline, candidate)

    metric = result["providers"]["gemini"]["metrics"][metric_name]

    assert metric["baseline"] == baseline_value

    assert metric["candidate"] == candidate_value

    assert metric["status"] == expected_status


# ==========================================================
# TOLERÂNCIAS E ESTABILIDADE
# ==========================================================


def test_small_quality_differences_are_classified_as_stable(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["pass_rate"] = 100.0
    candidate_metrics["pass_rate"] = 99.7

    baseline_metrics["average_score"] = 1.0
    candidate_metrics["average_score"] = 0.998

    baseline_metrics["row_accuracy"] = 100.0
    candidate_metrics["row_accuracy"] = 99.7

    baseline_metrics["cell_accuracy"] = 100.0
    candidate_metrics["cell_accuracy"] = 99.7

    result = comparator.compare(baseline, candidate)

    metrics = result["providers"]["gemini"]["metrics"]

    assert metrics["pass_rate"]["status"] == "stable"

    assert metrics["average_score"]["status"] == "stable"

    assert metrics["row_accuracy"]["status"] == "stable"

    assert metrics["cell_accuracy"]["status"] == "stable"


def test_latency_increase_below_threshold_is_stable(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["average_duration_ms"] = 100.0
    candidate_metrics["average_duration_ms"] = 105.0

    result = comparator.compare(baseline, candidate)

    assert (
        result["providers"]["gemini"]["metrics"]["average_duration_ms"]["status"]
        == "stable"
    )


def test_latency_increase_above_threshold_is_regression(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["average_duration_ms"] = 100.0
    candidate_metrics["average_duration_ms"] = 125.0

    result = comparator.compare(baseline, candidate)

    assert (
        result["providers"]["gemini"]["metrics"]["average_duration_ms"]["status"]
        == "regression"
    )


def test_comparator_accepts_custom_thresholds(
    baseline,
    candidate,
):
    comparator = BenchmarkComparator(
        thresholds={
            "pass_rate": 5.0,
            "average_score": 0.05,
        }
    )

    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["pass_rate"] = 100.0
    candidate_metrics["pass_rate"] = 95.0

    baseline_metrics["average_score"] = 1.0
    candidate_metrics["average_score"] = 0.95

    result = comparator.compare(baseline, candidate)

    metrics = result["providers"]["gemini"]["metrics"]

    assert metrics["pass_rate"]["status"] == "stable"

    assert metrics["average_score"]["status"] == "stable"


# ==========================================================
# TOKENS, LATÊNCIA E REQUISIÇÕES
# ==========================================================


@pytest.mark.parametrize(
    (
        "metric_path",
        "baseline_value",
        "candidate_value",
        "expected_status",
    ),
    [
        ("tokens.total", 1000, 1500, "regression"),
        ("tokens.total", 1500, 1000, "improvement"),
        ("requests.failed", 0, 2, "regression"),
        ("requests.failed", 2, 0, "improvement"),
        ("requests.retries", 0, 2, "regression"),
        ("requests.retries", 2, 0, "improvement"),
        ("latency.average_request_ms", 100.0, 150.0, "regression"),
        ("latency.average_request_ms", 150.0, 100.0, "improvement"),
    ],
)
def test_comparison_classifies_llm_usage_correctly(
    comparator,
    baseline,
    candidate,
    metric_path,
    baseline_value,
    candidate_value,
    expected_status,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    section, field = metric_path.split(".")

    baseline_metrics["llm_usage"][section][field] = baseline_value

    candidate_metrics["llm_usage"][section][field] = candidate_value

    result = comparator.compare(baseline, candidate)

    metric = result["providers"]["gemini"]["metrics"][f"llm_usage.{metric_path}"]

    assert metric["baseline"] == baseline_value

    assert metric["candidate"] == candidate_value

    assert metric["status"] == expected_status


def test_comparison_does_not_use_cost_estimate_as_real_billing(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["llm_usage"]["cost_estimate_usd"] = 0.0

    candidate_metrics["llm_usage"]["cost_estimate_usd"] = 500.0

    result = comparator.compare(baseline, candidate)

    metrics = result["providers"]["gemini"]["metrics"]

    assert "llm_usage.cost_estimate_usd" not in metrics


def test_comparison_calculates_metric_deltas(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["pass_rate"] = 90.0

    candidate_metrics["pass_rate"] = 80.0

    baseline_metrics["average_duration_ms"] = 100.0

    candidate_metrics["average_duration_ms"] = 120.0

    result = comparator.compare(baseline, candidate)

    metrics = result["providers"]["gemini"]["metrics"]

    assert metrics["pass_rate"]["delta"] == pytest.approx(-10.0)

    assert metrics["pass_rate"]["delta_pct"] == pytest.approx(-11.1111, abs=0.001)

    assert metrics["average_duration_ms"]["delta"] == pytest.approx(20.0)

    assert metrics["average_duration_ms"]["delta_pct"] == pytest.approx(20.0)


# ==========================================================
# MÉTRICAS AUSENTES OU INVÁLIDAS
# ==========================================================


def test_missing_metric_is_not_comparable(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["row_accuracy"] = None

    candidate_metrics["row_accuracy"] = 95.0

    result = comparator.compare(baseline, candidate)

    metric = result["providers"]["gemini"]["metrics"]["row_accuracy"]

    assert metric["status"] == "not_comparable"

    assert metric["delta"] is None


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        float("nan"),
        float("inf"),
        -1.0,
        "100",
    ],
)
def test_invalid_metric_values_are_not_comparable(
    comparator,
    baseline,
    candidate,
    invalid_value,
):
    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics["pass_rate"] = invalid_value

    result = comparator.compare(baseline, candidate)

    metric = result["providers"]["gemini"]["metrics"]["pass_rate"]

    assert metric["status"] == "not_comparable"

    assert metric["delta"] is None


def test_zero_baseline_has_no_percentage_delta(
    comparator,
    baseline,
    candidate,
):
    baseline_metrics = baseline["comparison"]["by_provider"]["gemini"]["metrics"]

    candidate_metrics = candidate["comparison"]["by_provider"]["gemini"]["metrics"]

    baseline_metrics["average_duration_ms"] = 0.0

    candidate_metrics["average_duration_ms"] = 10.0

    result = comparator.compare(baseline, candidate)

    metric = result["providers"]["gemini"]["metrics"]["average_duration_ms"]

    assert metric["delta"] == 10.0

    assert metric["delta_pct"] is None


# ==========================================================
# FALHAS DE EXECUÇÃO
# ==========================================================


def test_candidate_provider_failure_is_regression(
    comparator,
    baseline,
    candidate,
):
    candidate["provider_runs"]["gemini"]["execution_status"] = "error"

    candidate["provider_runs"]["gemini"]["report"] = None

    candidate["comparison"]["by_provider"]["gemini"]["execution_status"] = "error"

    candidate["comparison"]["by_provider"]["gemini"]["metrics"] = None

    result = comparator.compare(baseline, candidate)

    assert result["providers"]["gemini"]["status"] == "regression"

    assert result["providers"]["gemini"]["execution"]["status"] == "regression"


def test_recovered_provider_execution_is_improvement(
    comparator,
    baseline,
    candidate,
):
    baseline["provider_runs"]["gemini"]["execution_status"] = "error"

    baseline["provider_runs"]["gemini"]["report"] = None

    baseline["comparison"]["by_provider"]["gemini"]["execution_status"] = "error"

    baseline["comparison"]["by_provider"]["gemini"]["metrics"] = None

    result = comparator.compare(baseline, candidate)

    assert result["providers"]["gemini"]["status"] == "improvement"

    assert result["providers"]["gemini"]["execution"]["status"] == "improvement"


# ==========================================================
# CLASSIFICAÇÃO CONSOLIDADA
# ==========================================================


def test_overall_status_is_regression_when_quality_deteriorates(
    comparator,
    baseline,
    candidate,
):
    candidate["comparison"]["by_provider"]["gemini"]["metrics"]["pass_rate"] = 80.0

    result = comparator.compare(baseline, candidate)

    assert result["summary"]["status"] == "regression"

    assert result["summary"]["regressions"] > 0


def test_overall_status_is_improvement_when_quality_increases(
    comparator,
    baseline,
    candidate,
):
    baseline["comparison"]["by_provider"]["gemini"]["metrics"]["pass_rate"] = 80.0

    result = comparator.compare(baseline, candidate)

    assert result["summary"]["status"] == "improvement"

    assert result["summary"]["improvements"] > 0


def test_overall_status_is_mixed_when_improvements_and_regressions_coexist(
    comparator,
    baseline,
    candidate,
):
    candidate["comparison"]["by_provider"]["gemini"]["metrics"]["pass_rate"] = 80.0

    candidate["comparison"]["by_provider"]["openai"]["metrics"]["pass_rate"] = 100.0

    baseline["comparison"]["by_provider"]["openai"]["metrics"]["pass_rate"] = 80.0

    result = comparator.compare(baseline, candidate)

    assert result["summary"]["status"] == "mixed"

    assert result["summary"]["regressions"] > 0

    assert result["summary"]["improvements"] > 0


def test_overall_status_is_not_comparable_when_no_models_match(
    comparator,
    baseline,
    candidate,
):
    candidate["provider_runs"]["gemini"]["report"]["model"] = "new-gemini-model"

    candidate["comparison"]["by_provider"]["gemini"]["model"] = "new-gemini-model"

    candidate["provider_runs"]["openai"]["report"]["model"] = "new-openai-model"

    candidate["comparison"]["by_provider"]["openai"]["model"] = "new-openai-model"

    result = comparator.compare(baseline, candidate)

    assert result["summary"]["status"] == "not_comparable"

    assert result["summary"]["providers_compared"] == 0


# ==========================================================
# VALIDAÇÃO DE ENTRADAS
# ==========================================================


@pytest.mark.parametrize(
    "invalid_report",
    [
        None,
        [],
        "benchmark",
        123,
    ],
)
def test_comparison_rejects_non_mapping_reports(
    comparator,
    invalid_report,
    baseline,
):
    with pytest.raises(TypeError):
        comparator.compare(invalid_report, baseline)

    with pytest.raises(TypeError):
        comparator.compare(baseline, invalid_report)


def test_comparison_rejects_missing_dataset(
    comparator,
    baseline,
    candidate,
):
    candidate.pop("dataset")

    with pytest.raises(ValueError):
        comparator.compare(baseline, candidate)


def test_comparison_rejects_missing_benchmark_id(
    comparator,
    baseline,
    candidate,
):
    candidate.pop("benchmark_id")

    with pytest.raises(ValueError):
        comparator.compare(baseline, candidate)


def test_comparison_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        BenchmarkComparator(
            thresholds={
                "pass_rate": -1.0,
            }
        )

    with pytest.raises(ValueError):
        BenchmarkComparator(
            thresholds={
                "average_score": float("nan"),
            }
        )
