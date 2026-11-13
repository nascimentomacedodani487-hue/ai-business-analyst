from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ai_analyst.evaluation.benchmark import BenchmarkRunner
from ai_analyst.evaluation.history import BenchmarkHistory
from ai_analyst.evaluation.models import EvaluationCase

# ==========================================================
# FÁBRICAS DE DADOS
# ==========================================================


def make_case(
    case_id: str = "total_orders",
    category: str = "volume",
    expected_analysis_mode: str | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"Pergunta de teste: {case_id}",
        reference_sql="SELECT COUNT(*) AS total_orders FROM orders",
        category=category,
        comparison_mode="exact",
        required_columns=("total_orders",),
        expected_analysis_mode=expected_analysis_mode,
    )


def make_llm_usage_snapshot(
    marker: str,
) -> dict[str, Any]:
    """Cria um snapshot opaco para testar sua preservação no benchmark."""

    return {
        "fixture_marker": marker,
        "requests": {
            "total": 10,
            "successful": 9,
            "failed": 1,
            "retries": 2,
        },
        "tokens": {
            "input": 1200,
            "output": 800,
            "thinking": 100,
            "total": 2100,
            "unreported": 0,
        },
        "latency": {
            "total_ms": 2500.0,
            "average_ms": 250.0,
        },
    }


def make_case_llm_metrics(
    provider: str,
    case_id: str,
    index: int,
) -> dict[str, Any]:
    """Cria métricas individuais e identificáveis por cenário e provedor."""

    multiplier = index + 1

    return {
        "analysis_id": f"{provider}:{case_id}:analysis",
        "provider": provider,
        "model": f"{provider}-model",
        "requests": {
            "total": multiplier,
            "successful": multiplier,
            "failed": 0,
            "retries": index,
        },
        "tokens": {
            "input": 100 * multiplier,
            "output": 50 * multiplier,
            "thinking": 10 * multiplier,
            "total": 160 * multiplier,
            "unreported": 0,
        },
        "latency": {
            "total_ms": 250.0 * multiplier,
            "average_ms": 250.0,
        },
    }


def make_report(
    provider: str,
    cases: list[EvaluationCase],
    *,
    model: str = "fake-model",
) -> dict[str, Any]:
    """Gera um relatório simulado do EvaluationRunner."""

    results = []

    for index, case in enumerate(cases):
        actual_mode = case.expected_analysis_mode or "single_query"

        matched_rows = 8 + index
        compared_rows = 10 + index

        matched_cells = 18 + (2 * index)
        compared_cells = 20 + (2 * index)

        metadata = {
            "category": case.category,
            "analysis_mode": actual_mode,
            "expected_analysis_mode": case.expected_analysis_mode,
            "analysis_mode_matches": (
                True if case.expected_analysis_mode is not None else None
            ),
            "matched_rows": matched_rows,
            "compared_rows": compared_rows,
            "row_accuracy": round(
                matched_rows / compared_rows * 100,
                2,
            ),
            "matched_cells": matched_cells,
            "compared_cells": compared_cells,
            "cell_accuracy": round(
                matched_cells / compared_cells * 100,
                2,
            ),
            "llm_metrics": make_case_llm_metrics(
                provider,
                case.case_id,
                index,
            ),
        }

        results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "status": "passed",
                "score": 1.0,
                "expected_rows": [
                    {"total_orders": 60_000},
                ],
                "actual_rows": [
                    {"total_orders": 60_000},
                ],
                "differences": [],
                "analysis_id": f"analysis-{index}",
                "provider": provider,
                "model": model,
                "duration_ms": 100.0,
                "cache_hit": False,
                "error": None,
                "metadata": metadata,
            }
        )

    total_cases = len(results)

    metrics_summary = {
        "total_cases": total_cases,
        "evaluated_cases": total_cases,
        "passed_cases": total_cases,
        "failed_cases": 0,
        "error_cases": 0,
        "skipped_cases": 0,
        "pass_rate": 100.0,
        "scored_cases": total_cases,
        "average_score": 1.0,
        "average_duration_ms": 100.0,
        "cache_observations": total_cases,
        "cache_hits": 0,
        "cache_misses": total_cases,
        "cache_hit_rate": 0.0,
        "row_accuracy": 100.0,
        "cell_accuracy": 100.0,
        "llm_usage": make_llm_usage_snapshot(
            f"{provider}:summary",
        ),
        "analysis_mode_adherence": {
            "expected_cases": sum(
                case.expected_analysis_mode is not None for case in cases
            ),
            "observed_cases": sum(
                case.expected_analysis_mode is not None for case in cases
            ),
            "unobserved_cases": 0,
            "matches": sum(case.expected_analysis_mode is not None for case in cases),
            "mismatches": 0,
            "rate": (
                100.0
                if any(case.expected_analysis_mode is not None for case in cases)
                else None
            ),
        },
    }

    by_category = {}

    for case in cases:
        category_results = [
            result
            for result in results
            if result["metadata"]["category"] == case.category
        ]

        category_count = len(category_results)

        by_category[case.category] = {
            "total_cases": category_count,
            "evaluated_cases": category_count,
            "passed_cases": category_count,
            "failed_cases": 0,
            "error_cases": 0,
            "skipped_cases": 0,
            "pass_rate": 100.0,
            "scored_cases": category_count,
            "average_score": 1.0,
            "average_duration_ms": 100.0,
            "cache_observations": category_count,
            "cache_hits": 0,
            "cache_misses": category_count,
            "cache_hit_rate": 0.0,
            "row_accuracy": 100.0,
            "cell_accuracy": 100.0,
            "llm_usage": make_llm_usage_snapshot(
                f"{provider}:{case.category}",
            ),
            "analysis_mode_adherence": {
                "expected_cases": sum(
                    item.expected_analysis_mode is not None
                    for item in cases
                    if item.category == case.category
                ),
                "observed_cases": sum(
                    item.expected_analysis_mode is not None
                    for item in cases
                    if item.category == case.category
                ),
                "unobserved_cases": 0,
                "matches": sum(
                    item.expected_analysis_mode is not None
                    for item in cases
                    if item.category == case.category
                ),
                "mismatches": 0,
                "rate": (
                    100.0
                    if any(
                        item.expected_analysis_mode is not None
                        for item in cases
                        if item.category == case.category
                    )
                    else None
                ),
            },
        }

    return {
        "run_id": f"run-{provider}",
        "provider": provider,
        "model": model,
        "summary": {
            "total_cases": total_cases,
            "evaluated_cases": total_cases,
            "passed_cases": total_cases,
            "failed_cases": 0,
            "error_cases": 0,
            "skipped_cases": 0,
            "pass_rate": 100.0,
            "duration_ms": 1000.0,
        },
        "results": results,
        "metadata": {
            "metrics": {
                "summary": metrics_summary,
                "by_category": by_category,
                "by_provider": {
                    provider: metrics_summary,
                },
                "by_model": {
                    f"{provider}/{model}": metrics_summary,
                },
            },
        },
    }


# ==========================================================
# RUNNER SIMULADO
# ==========================================================


class FakeRunner:
    """Simula EvaluationRunner sem chamadas reais de IA."""

    def __init__(
        self,
        provider_name: str,
        report: dict[str, Any] | None = None,
        error: Exception | None = None,
        executions: list[dict[str, Any]] | None = None,
    ):
        self.provider_name = provider_name
        self.report = report
        self.error = error
        self.executions = executions if executions is not None else []

    def run(
        self,
        cases: list[EvaluationCase],
        run_id: str,
    ) -> dict[str, Any]:
        self.executions.append(
            {
                "provider": self.provider_name,
                "run_id": run_id,
                "cases": list(cases),
            }
        )

        if self.error is not None:
            raise self.error

        return self.report


def make_benchmark(
    providers: list[str],
    cases: list[EvaluationCase],
    executions: list[dict[str, Any]] | None = None,
    errors: dict[str, Exception] | None = None,
    history_store: BenchmarkHistory | None = None,
) -> BenchmarkRunner:
    """Cria benchmark com runners simulados."""

    errors = errors or {}

    def runner_factory(
        provider_name: str,
        dataset_path=None,
    ):
        return FakeRunner(
            provider_name=provider_name,
            report=make_report(
                provider_name,
                cases,
            ),
            error=errors.get(provider_name),
            executions=executions,
        )

    return BenchmarkRunner(
        providers=providers,
        runner_factory=runner_factory,
        history_store=history_store,
    )


# ==========================================================
# INICIALIZAÇÃO E VALIDAÇÃO
# ==========================================================


@pytest.mark.parametrize(
    ("providers", "expected"),
    [
        (
            ["gemini", "openai"],
            ("gemini", "openai"),
        ),
        (
            [" GEMINI ", " OPENAI ", "ANTHROPIC"],
            ("gemini", "openai", "anthropic"),
        ),
    ],
)
def test_benchmark_normalizes_supported_providers(
    providers,
    expected,
):
    benchmark = BenchmarkRunner(
        providers=providers,
        runner_factory=lambda **kwargs: None,
    )

    assert benchmark.providers == expected


def test_benchmark_uses_all_supported_providers_by_default():
    benchmark = BenchmarkRunner(
        runner_factory=lambda **kwargs: None,
    )

    assert set(benchmark.providers) == {
        "gemini",
        "openai",
        "anthropic",
    }


def test_benchmark_requires_at_least_two_providers():
    with pytest.raises(
        ValueError,
        match="pelo menos dois provedores",
    ):
        BenchmarkRunner(
            providers=["gemini"],
        )


def test_benchmark_rejects_string_as_provider_collection():
    with pytest.raises(
        TypeError,
        match="não uma string individual",
    ):
        BenchmarkRunner(
            providers="gemini",
        )


def test_benchmark_rejects_non_string_provider():
    with pytest.raises(
        TypeError,
        match="devem ser strings",
    ):
        BenchmarkRunner(
            providers=["gemini", 123],
        )


def test_benchmark_rejects_empty_provider():
    with pytest.raises(
        ValueError,
        match="não pode ser vazio",
    ):
        BenchmarkRunner(
            providers=["gemini", ""],
        )


def test_benchmark_rejects_unsupported_provider():
    with pytest.raises(
        ValueError,
        match="Provedor não suportado",
    ):
        BenchmarkRunner(
            providers=["gemini", "invalid"],
        )


def test_benchmark_rejects_duplicate_providers():
    with pytest.raises(
        ValueError,
        match="Provedor duplicado",
    ):
        BenchmarkRunner(
            providers=["gemini", " GEMINI "],
        )


# ==========================================================
# VALIDAÇÃO DOS CENÁRIOS
# ==========================================================


def test_benchmark_requires_at_least_one_case():
    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=lambda **kwargs: None,
    )

    with pytest.raises(
        ValueError,
        match="pelo menos um cenário",
    ):
        benchmark.run(cases=[])


def test_benchmark_rejects_invalid_case_objects():
    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=lambda **kwargs: None,
    )

    with pytest.raises(
        TypeError,
        match="EvaluationCase",
    ):
        benchmark.run(cases=["invalid"])


def test_benchmark_rejects_duplicate_case_ids():
    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=lambda **kwargs: None,
    )

    case = make_case()

    with pytest.raises(
        ValueError,
        match="identificadores.*duplicados",
    ):
        benchmark.run(
            cases=[
                case,
                case,
            ],
        )


# ==========================================================
# FINGERPRINT DO DATASET
# ==========================================================


def test_dataset_fingerprint_is_sha256():
    cases = [
        make_case(),
    ]

    fingerprint = BenchmarkRunner._calculate_dataset_fingerprint(cases)

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 64

    assert all(character in "0123456789abcdef" for character in fingerprint)


def test_dataset_fingerprint_is_deterministic():
    cases = [
        make_case(),
        make_case(
            case_id="delivery",
            category="delivery",
        ),
    ]

    first = BenchmarkRunner._calculate_dataset_fingerprint(cases)
    second = BenchmarkRunner._calculate_dataset_fingerprint(cases)

    assert first == second


def test_dataset_fingerprint_changes_when_case_changes():
    original = make_case()

    modified = replace(
        original,
        question="Pergunta modificada.",
    )

    first = BenchmarkRunner._calculate_dataset_fingerprint([original])
    second = BenchmarkRunner._calculate_dataset_fingerprint([modified])

    assert first != second


# ==========================================================
# SNAPSHOT DE MÉTRICAS
# ==========================================================


def test_metric_snapshot_combines_run_and_metric_summaries():
    """Garante a preservação das métricas detalhadas."""

    llm_usage = {
        "requests": {
            "total": 10,
            "successful": 9,
            "failed": 1,
            "retries": 2,
        },
        "tokens": {
            "input": 1200,
            "output": 800,
            "thinking": 100,
            "total": 2100,
            "unreported": 0,
        },
    }

    report = {
        "summary": {
            "total_cases": 4,
            "error_cases": 1,
            "skipped_cases": 1,
        },
        "metadata": {
            "metrics": {
                "summary": {
                    "total_cases": 10,
                    "error_cases": 0,
                    "skipped_cases": 0,
                    "pass_rate": 50.0,
                    "average_score": 0.85,
                    "average_duration_ms": 250.0,
                    "cache_hit_rate": 25.0,
                    "row_accuracy": 91.25,
                    "cell_accuracy": 96.5,
                    "llm_usage": llm_usage,
                    "analysis_mode_adherence": {
                        "expected_cases": 2,
                        "observed_cases": 2,
                        "unobserved_cases": 0,
                        "matches": 1,
                        "mismatches": 1,
                        "rate": 50.0,
                    },
                },
            },
        },
    }

    snapshot = BenchmarkRunner._metric_snapshot(report)

    assert snapshot["total_cases"] == 4
    assert snapshot["error_cases"] == 1
    assert snapshot["skipped_cases"] == 1

    assert snapshot["pass_rate"] == 50.0
    assert snapshot["average_score"] == 0.85
    assert snapshot["average_duration_ms"] == 250.0
    assert snapshot["cache_hit_rate"] == 25.0

    assert snapshot["error_rate"] == 25.0
    assert snapshot["skipped_rate"] == 25.0

    assert snapshot["row_accuracy"] == 91.25
    assert snapshot["cell_accuracy"] == 96.5
    assert snapshot["llm_usage"] == llm_usage

    assert snapshot["analysis_mode_adherence"] == {
        "expected_cases": 2,
        "observed_cases": 2,
        "unobserved_cases": 0,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }


def test_metric_snapshot_prefers_run_summary_for_new_metrics():
    report = {
        "summary": {
            "row_accuracy": 80.0,
            "cell_accuracy": 85.0,
            "llm_usage": {
                "source": "run_summary",
            },
        },
        "metadata": {
            "metrics": {
                "summary": {
                    "row_accuracy": 95.0,
                    "cell_accuracy": 98.0,
                    "llm_usage": {
                        "source": "metrics_summary",
                    },
                },
            },
        },
    }

    snapshot = BenchmarkRunner._metric_snapshot(report)

    assert snapshot["row_accuracy"] == 80.0
    assert snapshot["cell_accuracy"] == 85.0
    assert snapshot["llm_usage"] == {
        "source": "run_summary",
    }


def test_metric_snapshot_supports_legacy_reports_without_new_metrics():
    report = {
        "summary": {
            "total_cases": 2,
            "evaluated_cases": 2,
            "passed_cases": 2,
            "failed_cases": 0,
            "error_cases": 0,
            "skipped_cases": 0,
            "pass_rate": 100.0,
        },
    }

    snapshot = BenchmarkRunner._metric_snapshot(report)

    assert snapshot["total_cases"] == 2
    assert snapshot["pass_rate"] == 100.0

    assert snapshot["row_accuracy"] is None
    assert snapshot["cell_accuracy"] is None
    assert snapshot["llm_usage"] is None


# ==========================================================
# EXECUÇÃO DO BENCHMARK
# ==========================================================


def test_benchmark_executes_all_providers_with_same_dataset():
    cases = [
        make_case(case_id="total_orders"),
        make_case(
            case_id="delivery",
            category="delivery",
        ),
    ]

    executions = []

    benchmark = make_benchmark(
        providers=[
            "gemini",
            "openai",
            "anthropic",
        ],
        cases=cases,
        executions=executions,
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="benchmark-test-001",
    )

    assert len(executions) == 3

    assert [item["provider"] for item in executions] == [
        "gemini",
        "openai",
        "anthropic",
    ]

    for execution in executions:
        assert execution["run_id"] == (f"benchmark-test-001-{execution['provider']}")
        assert execution["cases"] == cases

    assert report["benchmark_id"] == "benchmark-test-001"
    assert report["configuration"]["same_dataset_for_all_providers"] is True
    assert report["summary"]["provider_count"] == 3
    assert report["summary"]["completed_providers"] == 3
    assert report["summary"]["failed_providers"] == 0
    assert report["summary"]["all_providers_completed"] is True


def test_benchmark_generates_unique_ids_when_not_provided():
    cases = [
        make_case(),
    ]

    executions = []

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        executions=executions,
    )

    report = benchmark.run(cases=cases)

    assert report["benchmark_id"]
    assert executions[0]["run_id"].startswith(report["benchmark_id"])


def test_benchmark_rejects_empty_benchmark_id():
    cases = [
        make_case(),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    with pytest.raises(
        ValueError,
        match="benchmark_id",
    ):
        benchmark.run(
            cases=cases,
            benchmark_id=" ",
        )


# ==========================================================
# COMPARAÇÃO ENTRE PROVEDORES
# ==========================================================


def test_benchmark_generates_provider_comparison():
    cases = [
        make_case(),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="comparison-test",
    )

    comparison = report["comparison"]["by_provider"]

    assert set(comparison) == {
        "gemini",
        "openai",
    }

    assert comparison["gemini"]["execution_status"] == "completed"
    assert comparison["openai"]["execution_status"] == "completed"

    assert comparison["gemini"]["metrics"]["pass_rate"] == 100.0
    assert comparison["openai"]["metrics"]["average_score"] == 1.0

    for provider in ("gemini", "openai"):
        provider_metrics = comparison[provider]["metrics"]

        assert provider_metrics["row_accuracy"] == 100.0
        assert provider_metrics["cell_accuracy"] == 100.0

        assert provider_metrics["llm_usage"] == make_llm_usage_snapshot(
            f"{provider}:summary",
        )


def test_benchmark_generates_category_comparison():
    cases = [
        make_case(
            case_id="total_orders",
            category="volume",
            expected_analysis_mode="single_query",
        ),
        make_case(
            case_id="monthly_orders",
            category="volume",
            expected_analysis_mode="single_query",
        ),
        make_case(
            case_id="delivery",
            category="delivery",
            expected_analysis_mode="multi_query",
        ),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    report = benchmark.run(cases=cases)

    category_comparison = report["comparison"]["by_category"]

    assert set(category_comparison) == {
        "volume",
        "delivery",
    }

    assert set(category_comparison["volume"]) == {
        "gemini",
        "openai",
    }

    for provider in ("gemini", "openai"):
        volume_metrics = category_comparison["volume"][provider]

        assert volume_metrics["total_cases"] == 2
        assert volume_metrics["pass_rate"] == 100.0
        assert volume_metrics["average_score"] == 1.0
        assert volume_metrics["average_duration_ms"] == 100.0

        assert volume_metrics["error_cases"] == 0
        assert volume_metrics["skipped_cases"] == 0
        assert volume_metrics["error_rate"] == 0.0
        assert volume_metrics["skipped_rate"] == 0.0

        assert volume_metrics["row_accuracy"] == 100.0
        assert volume_metrics["cell_accuracy"] == 100.0

        assert volume_metrics["llm_usage"] == make_llm_usage_snapshot(
            f"{provider}:volume",
        )

        adherence = volume_metrics["analysis_mode_adherence"]

        assert adherence["expected_cases"] == 2
        assert adherence["observed_cases"] == 2
        assert adherence["matches"] == 2
        assert adherence["mismatches"] == 0
        assert adherence["rate"] == 100.0

    for provider in ("gemini", "openai"):
        delivery_metrics = category_comparison["delivery"][provider]

        assert delivery_metrics["total_cases"] == 1
        assert delivery_metrics["pass_rate"] == 100.0
        assert delivery_metrics["average_score"] == 1.0
        assert delivery_metrics["average_duration_ms"] == 100.0

        assert delivery_metrics["error_cases"] == 0
        assert delivery_metrics["skipped_cases"] == 0
        assert delivery_metrics["error_rate"] == 0.0
        assert delivery_metrics["skipped_rate"] == 0.0

        assert delivery_metrics["row_accuracy"] == 100.0
        assert delivery_metrics["cell_accuracy"] == 100.0

        assert delivery_metrics["llm_usage"] == make_llm_usage_snapshot(
            f"{provider}:delivery",
        )

        adherence = delivery_metrics["analysis_mode_adherence"]

        assert adherence["expected_cases"] == 1
        assert adherence["observed_cases"] == 1
        assert adherence["matches"] == 1
        assert adherence["mismatches"] == 0
        assert adherence["rate"] == 100.0


# ==========================================================
# COMPARAÇÃO DETALHADA POR CENÁRIO
# ==========================================================


def test_benchmark_generates_case_comparison():
    cases = [
        make_case(
            case_id="operational_overview",
            category="operational_overview",
            expected_analysis_mode="multi_query",
        ),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    report = benchmark.run(cases=cases)

    overview = report["comparison"]["by_case"]["operational_overview"]

    assert overview["expected_analysis_mode"] == "multi_query"

    assert overview["providers"]["gemini"]["status"] == "passed"
    assert overview["providers"]["openai"]["status"] == "passed"

    assert overview["providers"]["gemini"]["analysis_mode"] == "multi_query"

    assert overview["providers"]["gemini"]["analysis_mode_matches"] is True

    for provider in ("gemini", "openai"):
        provider_result = overview["providers"][provider]

        assert provider_result["matched_rows"] == 8
        assert provider_result["compared_rows"] == 10
        assert provider_result["row_accuracy"] == 80.0

        assert provider_result["matched_cells"] == 18
        assert provider_result["compared_cells"] == 20
        assert provider_result["cell_accuracy"] == 90.0

    assert overview["providers"]["gemini"]["llm_metrics"] == (
        make_case_llm_metrics("gemini", "operational_overview", 0)
    )

    assert overview["providers"]["openai"]["llm_metrics"] == (
        make_case_llm_metrics("openai", "operational_overview", 0)
    )


def test_case_comparison_supports_legacy_results_without_quality_metrics():
    cases = [
        make_case(),
    ]

    def runner_factory(
        provider_name: str,
        dataset_path=None,
    ):
        report = make_report(
            provider_name,
            cases,
        )

        metadata = report["results"][0]["metadata"]

        metadata.pop("matched_rows")
        metadata.pop("compared_rows")
        metadata.pop("row_accuracy")
        metadata.pop("matched_cells")
        metadata.pop("compared_cells")
        metadata.pop("cell_accuracy")
        metadata.pop("llm_metrics")

        return FakeRunner(
            provider_name=provider_name,
            report=report,
        )

    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=runner_factory,
    )

    report = benchmark.run(cases=cases)

    providers = report["comparison"]["by_case"]["total_orders"]["providers"]

    for provider in ("gemini", "openai"):
        result = providers[provider]

        assert result["status"] == "passed"

        assert result["matched_rows"] is None
        assert result["compared_rows"] is None
        assert result["row_accuracy"] is None

        assert result["matched_cells"] is None
        assert result["compared_cells"] is None
        assert result["cell_accuracy"] is None

        assert result["llm_metrics"] is None


def test_benchmark_preserves_llm_metrics_from_multiple_scenarios():
    cases = [
        make_case(
            case_id="total_orders",
            category="volume",
        ),
        make_case(
            case_id="delivery",
            category="delivery",
        ),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    report = benchmark.run(cases=cases)

    by_case = report["comparison"]["by_case"]

    orders = by_case["total_orders"]["providers"]
    delivery = by_case["delivery"]["providers"]

    gemini_orders_metrics = orders["gemini"]["llm_metrics"]
    gemini_delivery_metrics = delivery["gemini"]["llm_metrics"]

    assert gemini_orders_metrics["analysis_id"] == ("gemini:total_orders:analysis")

    assert gemini_delivery_metrics["analysis_id"] == ("gemini:delivery:analysis")

    assert gemini_orders_metrics["tokens"]["total"] == 160
    assert gemini_delivery_metrics["tokens"]["total"] == 320

    assert gemini_orders_metrics is not gemini_delivery_metrics

    openai_orders_metrics = orders["openai"]["llm_metrics"]
    openai_delivery_metrics = delivery["openai"]["llm_metrics"]

    assert openai_orders_metrics["analysis_id"] == ("openai:total_orders:analysis")

    assert openai_delivery_metrics["analysis_id"] == ("openai:delivery:analysis")

    assert openai_orders_metrics["tokens"]["total"] == 160
    assert openai_delivery_metrics["tokens"]["total"] == 320

    assert gemini_orders_metrics["provider"] == "gemini"
    assert openai_orders_metrics["provider"] == "openai"

    assert gemini_orders_metrics is not openai_orders_metrics

    assert (
        gemini_orders_metrics["analysis_id"] != (openai_orders_metrics["analysis_id"])
    )


def test_benchmark_preserves_case_specific_llm_latency_and_retries():
    cases = [
        make_case(case_id="total_orders"),
        make_case(case_id="delivery"),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    report = benchmark.run(cases=cases)

    by_case = report["comparison"]["by_case"]

    for provider in ("gemini", "openai"):
        first_metrics = by_case["total_orders"]["providers"][provider]["llm_metrics"]

        second_metrics = by_case["delivery"]["providers"][provider]["llm_metrics"]

        assert first_metrics["requests"]["total"] == 1
        assert first_metrics["requests"]["retries"] == 0
        assert first_metrics["tokens"]["input"] == 100
        assert first_metrics["tokens"]["output"] == 50
        assert first_metrics["latency"]["total_ms"] == 250.0

        assert second_metrics["requests"]["total"] == 2
        assert second_metrics["requests"]["retries"] == 1
        assert second_metrics["tokens"]["input"] == 200
        assert second_metrics["tokens"]["output"] == 100
        assert second_metrics["latency"]["total_ms"] == 500.0


# ==========================================================
# ISOLAMENTO DE FALHAS
# ==========================================================


def test_benchmark_continues_when_one_provider_fails():
    cases = [
        make_case(),
    ]

    executions = []

    benchmark = make_benchmark(
        providers=[
            "gemini",
            "openai",
            "anthropic",
        ],
        cases=cases,
        executions=executions,
        errors={
            "openai": RuntimeError("Falha simulada no provedor."),
        },
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="failure-isolation-test",
    )

    assert len(executions) == 3

    assert report["summary"]["completed_providers"] == 2
    assert report["summary"]["failed_providers"] == 1
    assert report["summary"]["all_providers_completed"] is False

    assert report["provider_runs"]["gemini"]["execution_status"] == "completed"
    assert report["provider_runs"]["openai"]["execution_status"] == "error"
    assert report["provider_runs"]["anthropic"]["execution_status"] == "completed"


def test_benchmark_records_provider_factory_failure():
    cases = [
        make_case(),
    ]

    initialized = []

    def runner_factory(
        provider_name: str,
        dataset_path=None,
    ):
        initialized.append(provider_name)

        if provider_name == "openai":
            raise RuntimeError("Falha simulada na inicialização.")

        return FakeRunner(
            provider_name=provider_name,
            report=make_report(
                provider_name,
                cases,
            ),
        )

    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=runner_factory,
    )

    report = benchmark.run(cases=cases)

    assert initialized == [
        "gemini",
        "openai",
    ]

    assert report["provider_runs"]["openai"]["execution_status"] == "error"


def test_case_comparison_marks_failed_provider_as_not_executed():
    cases = [
        make_case(),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        errors={
            "openai": RuntimeError("Falha simulada."),
        },
    )

    report = benchmark.run(cases=cases)

    providers = report["comparison"]["by_case"]["total_orders"]["providers"]

    assert providers["gemini"]["status"] == "passed"

    assert providers["gemini"]["llm_metrics"] == (
        make_case_llm_metrics("gemini", "total_orders", 0)
    )

    failed_result = providers["openai"]

    assert failed_result["status"] == "not_executed"

    assert failed_result["matched_rows"] is None
    assert failed_result["compared_rows"] is None
    assert failed_result["row_accuracy"] is None

    assert failed_result["matched_cells"] is None
    assert failed_result["compared_cells"] is None
    assert failed_result["cell_accuracy"] is None

    assert failed_result["llm_metrics"] is None

    assert failed_result["error"] == "Falha simulada."


# ==========================================================
# RESULTADOS AUSENTES E RELATÓRIOS INVÁLIDOS
# ==========================================================


def test_benchmark_marks_missing_case_result():
    cases = [
        make_case(),
    ]

    empty_report = make_report(
        "openai",
        [],
    )

    def runner_factory(
        provider_name: str,
        dataset_path=None,
    ):
        return FakeRunner(
            provider_name=provider_name,
            report=empty_report,
        )

    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=runner_factory,
    )

    report = benchmark.run(cases=cases)

    openai_result = report["comparison"]["by_case"]["total_orders"]["providers"][
        "openai"
    ]

    assert openai_result["status"] == "missing_result"
    assert openai_result["score"] is None

    assert openai_result["matched_rows"] is None
    assert openai_result["compared_rows"] is None
    assert openai_result["row_accuracy"] is None

    assert openai_result["matched_cells"] is None
    assert openai_result["compared_cells"] is None
    assert openai_result["cell_accuracy"] is None

    assert openai_result["llm_metrics"] is None


def test_benchmark_rejects_invalid_report_contract():
    cases = [
        make_case(),
    ]

    def runner_factory(
        provider_name: str,
        dataset_path=None,
    ):
        return FakeRunner(
            provider_name=provider_name,
            report="invalid-report",
        )

    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=runner_factory,
    )

    report = benchmark.run(cases=cases)

    assert report["summary"]["failed_providers"] == 2
    assert report["provider_runs"]["gemini"]["error_type"] == "TypeError"


# ==========================================================
# PERSISTÊNCIA INTEGRADA COM BENCHMARK HISTORY
# ==========================================================


def test_benchmark_runner_persists_report_when_history_is_injected(tmp_path):
    cases = [
        make_case(),
    ]

    history = BenchmarkHistory(
        tmp_path / "integrated_benchmark_history.sqlite3",
    )

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        history_store=history,
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="integrated-001",
    )

    stored_report = history.get_benchmark("integrated-001")

    assert stored_report is not None
    assert stored_report == report

    assert history.count_benchmarks() == 1

    summaries = history.list_benchmarks()

    assert summaries["pagination"]["total"] == 1
    assert summaries["items"][0]["benchmark_id"] == "integrated-001"


def test_benchmark_runner_persists_partial_provider_failures(tmp_path):
    cases = [
        make_case(),
    ]

    history = BenchmarkHistory(
        tmp_path / "partial_benchmark_history.sqlite3",
    )

    benchmark = make_benchmark(
        providers=["gemini", "openai", "anthropic"],
        cases=cases,
        history_store=history,
        errors={
            "openai": RuntimeError("Falha simulada."),
        },
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="partial-001",
    )

    stored_report = history.get_benchmark("partial-001")

    assert stored_report == report

    assert stored_report["summary"]["completed_providers"] == 2
    assert stored_report["summary"]["failed_providers"] == 1

    assert stored_report["provider_runs"]["openai"]["execution_status"] == "error"

    summary = history.list_benchmarks()["items"][0]

    assert summary["summary"]["completed_providers"] == 2
    assert summary["summary"]["failed_providers"] == 1
    assert summary["summary"]["all_providers_completed"] is False


def test_benchmark_runner_does_not_persist_without_history_store():
    cases = [
        make_case(),
    ]

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
    )

    assert benchmark.history_store is None

    report = benchmark.run(
        cases=cases,
        benchmark_id="without-history-001",
    )

    assert report["benchmark_id"] == "without-history-001"

    assert benchmark.history_store is None


def test_benchmark_runner_propagates_history_persistence_errors(
    tmp_path,
    monkeypatch,
):
    cases = [
        make_case(),
    ]

    history = BenchmarkHistory(
        tmp_path / "failing_benchmark_history.sqlite3",
    )

    def fail_save(report):
        raise RuntimeError("Falha simulada no armazenamento.")

    monkeypatch.setattr(
        history,
        "save_benchmark",
        fail_save,
    )

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        history_store=history,
    )

    with pytest.raises(
        RuntimeError,
        match="Falha simulada no armazenamento",
    ):
        benchmark.run(
            cases=cases,
            benchmark_id="persistence-error-001",
        )

    assert history.count_benchmarks() == 0


def test_benchmark_runner_does_not_overwrite_existing_history_record(tmp_path):
    cases = [
        make_case(),
    ]

    history = BenchmarkHistory(
        tmp_path / "immutable_benchmark_history.sqlite3",
    )

    benchmark = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        history_store=history,
    )

    first_report = benchmark.run(
        cases=cases,
        benchmark_id="immutable-001",
    )

    stored_before = history.get_benchmark("immutable-001")

    assert stored_before == first_report

    with pytest.raises(
        ValueError,
        match="já está armazenado",
    ):
        benchmark.run(
            cases=cases,
            benchmark_id="immutable-001",
        )

    stored_after = history.get_benchmark("immutable-001")

    assert stored_after == stored_before
    assert history.count_benchmarks() == 1
