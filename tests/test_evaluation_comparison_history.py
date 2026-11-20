"""Testes de integração entre BenchmarkComparator e BenchmarkHistory."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ai_analyst.evaluation.comparison import BenchmarkComparator
from ai_analyst.evaluation.history import BenchmarkHistory

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
    benchmark_id: str,
    created_at: str,
    gemini_metrics: dict[str, Any] | None = None,
    openai_metrics: dict[str, Any] | None = None,
    gemini_model: str = "gemini-2.5-flash",
    openai_model: str = "gpt-4.1-mini",
) -> dict[str, Any]:
    """Cria um relatório válido para persistência histórica."""

    providers = ("gemini", "openai")

    models = {
        "gemini": gemini_model,
        "openai": openai_model,
    }

    metrics = {
        "gemini": gemini_metrics or make_provider_metrics(),
        "openai": openai_metrics or make_provider_metrics(),
    }

    provider_runs = {}

    by_provider = {}

    for provider in providers:
        model = models[provider]

        provider_runs[provider] = {
            "execution_status": "completed",
            "report": {
                "run_id": f"{benchmark_id}-{provider}",
                "provider": provider,
                "model": model,
                "summary": {
                    "total_cases": 2,
                    "evaluated_cases": 2,
                    "passed_cases": 2,
                    "failed_cases": 0,
                    "error_cases": 0,
                    "skipped_cases": 0,
                    "pass_rate": 100.0,
                    "duration_ms": 200.0,
                },
                "results": [],
            },
            "error": None,
            "error_type": None,
        }

        by_provider[provider] = {
            "execution_status": "completed",
            "model": model,
            "run_id": f"{benchmark_id}-{provider}",
            "metrics": metrics[provider],
        }

    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "dataset": {
            "version": 1,
            "source": None,
            "case_count": 2,
            "case_ids": [
                "case-1",
                "case-2",
            ],
            "sha256": "a" * 64,
        },
        "configuration": {
            "providers": list(providers),
            "provider_count": len(providers),
            "scenario_count": 2,
            "execution_strategy": "sequential",
            "same_dataset_for_all_providers": True,
        },
        "summary": {
            "provider_count": 2,
            "completed_providers": 2,
            "failed_providers": 0,
            "all_providers_completed": True,
        },
        "provider_runs": provider_runs,
        "comparison": {
            "by_provider": by_provider,
            "by_category": {},
            "by_case": {},
        },
    }


# ==========================================================
# FIXTURES
# ==========================================================


@pytest.fixture
def history(tmp_path: Path) -> BenchmarkHistory:
    """Cria um histórico SQLite isolado para cada teste."""

    database_path = tmp_path / "benchmark_history.sqlite3"

    return BenchmarkHistory(database_path)


@pytest.fixture
def comparator() -> BenchmarkComparator:
    """Cria o comparador padrão."""

    return BenchmarkComparator()


@pytest.fixture
def persisted_benchmarks(
    history: BenchmarkHistory,
) -> dict[str, dict[str, Any]]:
    """Persiste dois benchmarks com o mesmo dataset."""

    baseline = make_benchmark_report(
        benchmark_id="baseline-001",
        created_at="2026-11-12T10:00:00+00:00",
    )

    candidate = make_benchmark_report(
        benchmark_id="candidate-001",
        created_at="2026-11-12T11:00:00+00:00",
    )

    history.save_benchmark(baseline)
    history.save_benchmark(candidate)

    return {
        "baseline": baseline,
        "candidate": candidate,
    }


# ==========================================================
# INTEGRAÇÃO COM SQLITE
# ==========================================================


def test_comparator_compares_reports_retrieved_from_sqlite(
    history,
    comparator,
    persisted_benchmarks,
):
    """Recupera dois relatórios reais do SQLite e compara seus resultados."""

    baseline = history.get_benchmark("baseline-001")

    candidate = history.get_benchmark("candidate-001")

    assert baseline is not None
    assert candidate is not None

    result = comparator.compare(
        baseline,
        candidate,
    )

    assert result["compatibility"]["comparable"] is True

    assert result["baseline"]["benchmark_id"] == "baseline-001"

    assert result["candidate"]["benchmark_id"] == "candidate-001"

    assert result["summary"]["status"] == "stable"

    assert result["summary"]["providers_compared"] == 2


def test_comparator_detects_regression_in_persisted_benchmark(
    history,
    comparator,
):
    """Uma queda de qualidade persistida deve ser identificada."""

    baseline = make_benchmark_report(
        benchmark_id="baseline-regression",
        created_at="2026-11-12T10:00:00+00:00",
    )

    candidate = make_benchmark_report(
        benchmark_id="candidate-regression",
        created_at="2026-11-12T11:00:00+00:00",
        gemini_metrics=make_provider_metrics(
            pass_rate=80.0,
            average_score=0.80,
            row_accuracy=90.0,
            cell_accuracy=90.0,
        ),
    )

    history.save_benchmark(baseline)
    history.save_benchmark(candidate)

    baseline_from_database = history.get_benchmark(
        "baseline-regression",
    )

    candidate_from_database = history.get_benchmark(
        "candidate-regression",
    )

    assert baseline_from_database is not None
    assert candidate_from_database is not None

    result = comparator.compare(
        baseline_from_database,
        candidate_from_database,
    )

    assert result["summary"]["status"] == "regression"

    assert result["providers"]["gemini"]["status"] == "regression"

    assert (
        result["providers"]["gemini"]["metrics"]["pass_rate"]["status"] == "regression"
    )

    assert result["providers"]["openai"]["status"] == "compared"


def test_comparator_detects_model_change_in_persisted_benchmarks(
    history,
    comparator,
):
    """Uma troca de modelo não deve ser tratada como regressão de qualidade."""

    baseline = make_benchmark_report(
        benchmark_id="baseline-model",
        created_at="2026-11-12T10:00:00+00:00",
    )

    candidate = make_benchmark_report(
        benchmark_id="candidate-model",
        created_at="2026-11-12T11:00:00+00:00",
        gemini_model="gemini-new-model",
    )

    history.save_benchmark(baseline)
    history.save_benchmark(candidate)

    baseline_from_database = history.get_benchmark(
        "baseline-model",
    )

    candidate_from_database = history.get_benchmark(
        "candidate-model",
    )

    assert baseline_from_database is not None
    assert candidate_from_database is not None

    result = comparator.compare(
        baseline_from_database,
        candidate_from_database,
    )

    assert result["providers"]["gemini"]["status"] == "model_changed"

    assert result["providers"]["gemini"]["metrics"] == {}

    assert result["providers"]["openai"]["status"] == "compared"


def test_comparison_does_not_modify_reports_retrieved_from_sqlite(
    history,
    comparator,
    persisted_benchmarks,
):
    """A comparação não deve alterar relatórios recuperados."""

    baseline = history.get_benchmark("baseline-001")

    candidate = history.get_benchmark("candidate-001")

    assert baseline is not None
    assert candidate is not None

    original_baseline = deepcopy(baseline)
    original_candidate = deepcopy(candidate)

    comparator.compare(
        baseline,
        candidate,
    )

    assert baseline == original_baseline
    assert candidate == original_candidate

    assert history.get_benchmark("baseline-001") == original_baseline

    assert history.get_benchmark("candidate-001") == original_candidate


def test_comparator_rejects_different_fingerprints_from_sqlite(
    history,
    comparator,
):
    """Relatórios persistidos com datasets diferentes são incompatíveis."""

    baseline = make_benchmark_report(
        benchmark_id="baseline-dataset",
        created_at="2026-11-12T10:00:00+00:00",
    )

    candidate = make_benchmark_report(
        benchmark_id="candidate-dataset",
        created_at="2026-11-12T11:00:00+00:00",
    )

    candidate["dataset"]["sha256"] = "b" * 64

    history.save_benchmark(baseline)
    history.save_benchmark(candidate)

    baseline_from_database = history.get_benchmark(
        "baseline-dataset",
    )

    candidate_from_database = history.get_benchmark(
        "candidate-dataset",
    )

    assert baseline_from_database is not None
    assert candidate_from_database is not None

    result = comparator.compare(
        baseline_from_database,
        candidate_from_database,
    )

    assert result["compatibility"]["comparable"] is False

    assert result["summary"]["status"] == "incompatible"

    assert "dataset_fingerprint_mismatch" in (result["compatibility"]["reasons"])


def test_history_list_can_find_benchmarks_used_in_comparison(
    history,
    persisted_benchmarks,
):
    """Os dois benchmarks persistidos devem aparecer na listagem histórica."""

    result = history.list_benchmarks(
        limit=10,
        offset=0,
    )

    assert result["pagination"]["total"] == 2

    benchmark_ids = {item["benchmark_id"] for item in result["items"]}

    assert benchmark_ids == {
        "baseline-001",
        "candidate-001",
    }


def test_comparison_uses_independent_benchmark_ids(
    history,
    comparator,
    persisted_benchmarks,
):
    """Não deve ser possível confundir a referência com a candidata."""

    baseline = history.get_benchmark("baseline-001")

    candidate = history.get_benchmark("candidate-001")

    assert baseline is not None
    assert candidate is not None

    result = comparator.compare(
        baseline,
        candidate,
    )

    assert result["baseline"]["benchmark_id"] != (result["candidate"]["benchmark_id"])
