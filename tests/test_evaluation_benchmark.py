from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ai_analyst.evaluation.benchmark import BenchmarkRunner
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

        metadata = {
            "category": case.category,
            "analysis_mode": actual_mode,
            "expected_analysis_mode": (case.expected_analysis_mode),
            "analysis_mode_matches": (
                True if case.expected_analysis_mode is not None else None
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
    """
    Garante que métricas detalhadas não sejam descartadas
    quando o relatório também possui um summary de execução.

    Os campos presentes nos dois summaries devem respeitar
    a precedência do summary principal da execução.
    """

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

    assert snapshot["analysis_mode_adherence"] == {
        "expected_cases": 2,
        "observed_cases": 2,
        "unobserved_cases": 0,
        "matches": 1,
        "mismatches": 1,
        "rate": 50.0,
    }


# ==========================================================
# EXECUÇÃO DO BENCHMARK
# ==========================================================


def test_benchmark_executes_all_providers_with_same_dataset():

    cases = [
        make_case(
            case_id="total_orders",
        ),
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

    report = benchmark.run(
        cases=cases,
    )

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

    # ------------------------------------------------------
    # Validação das métricas da categoria de volume
    # ------------------------------------------------------

    for provider in ("gemini", "openai"):
        # As métricas são retornadas diretamente,
        # sem uma camada adicional chamada "metrics".
        volume_metrics = category_comparison["volume"][provider]

        assert volume_metrics["total_cases"] == 2
        assert volume_metrics["pass_rate"] == 100.0
        assert volume_metrics["average_score"] == 1.0
        assert volume_metrics["average_duration_ms"] == 100.0

        assert volume_metrics["error_cases"] == 0
        assert volume_metrics["skipped_cases"] == 0
        assert volume_metrics["error_rate"] == 0.0
        assert volume_metrics["skipped_rate"] == 0.0

        adherence = volume_metrics["analysis_mode_adherence"]

        assert adherence["expected_cases"] == 2
        assert adherence["observed_cases"] == 2
        assert adherence["matches"] == 2
        assert adherence["mismatches"] == 0
        assert adherence["rate"] == 100.0

    # ------------------------------------------------------
    # Validação das métricas da categoria de delivery
    # ------------------------------------------------------

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

        adherence = delivery_metrics["analysis_mode_adherence"]

        assert adherence["expected_cases"] == 1
        assert adherence["observed_cases"] == 1
        assert adherence["matches"] == 1
        assert adherence["mismatches"] == 0
        assert adherence["rate"] == 100.0


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

    case_comparison = report["comparison"]["by_case"]

    overview = case_comparison["operational_overview"]

    assert overview["expected_analysis_mode"] == "multi_query"

    assert overview["providers"]["gemini"]["status"] == "passed"

    assert overview["providers"]["openai"]["status"] == "passed"

    assert overview["providers"]["gemini"]["analysis_mode"] == "multi_query"

    assert overview["providers"]["gemini"]["analysis_mode_matches"] is True


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
