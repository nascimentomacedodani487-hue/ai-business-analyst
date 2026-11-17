"""Testes de repetibilidade e estabilidade dos benchmarks."""

from collections import defaultdict
from statistics import stdev

import pytest

from ai_analyst.evaluation.benchmark import BenchmarkRunner
from ai_analyst.evaluation.models import EvaluationCase

# ==========================================================
# FÁBRICAS DE DADOS
# ==========================================================


def make_case(
    case_id: str = "total_orders",
    category: str = "volume",
) -> EvaluationCase:
    """Cria um cenário SQL de avaliação."""

    return EvaluationCase(
        case_id=case_id,
        question=f"Pergunta de teste: {case_id}",
        reference_sql="SELECT COUNT(*) AS total_orders FROM orders",
        category=category,
        comparison_mode="exact",
        required_columns=("total_orders",),
    )


def make_report(
    provider: str,
    run_id: str,
    *,
    pass_rate: float = 100.0,
    average_score: float = 1.0,
    duration_ms: float = 100.0,
    cache_hit_rate: float = 0.0,
    estimated_cost_usd: float = 0.001,
    input_tokens: int = 100,
    output_tokens: int = 50,
    status: str = "passed",
    case_score: float = 1.0,
    case_duration_ms: float = 100.0,
    cache_hit: bool = False,
    include_case: bool = True,
) -> dict:
    """Cria um relatório simulado com métricas variáveis."""

    summary = {
        "total_cases": 1,
        "evaluated_cases": 1,
        "passed_cases": 1 if status == "passed" else 0,
        "failed_cases": 1 if status == "failed" else 0,
        "error_cases": 1 if status == "error" else 0,
        "skipped_cases": 0,
        "pass_rate": pass_rate,
        "scored_cases": 1,
        "average_score": average_score,
        "average_duration_ms": duration_ms,
        "cache_observations": 1,
        "cache_hits": 1 if cache_hit else 0,
        "cache_misses": 0 if cache_hit else 1,
        "cache_hit_rate": cache_hit_rate,
        "llm_usage": {
            "requests": {
                "total": 1,
                "successful": 1,
                "failed": 0,
                "retries": 0,
            },
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "thinking": 0,
                "total": input_tokens + output_tokens,
                "unreported": 0,
            },
            "latency": {
                "total_ms": duration_ms,
                "average_request_ms": duration_ms,
                "average_case_ms": duration_ms,
                "measured_cases": 1,
            },
            "token_cost_estimate": {
                "basis": "standard_public_list_price",
                "currency": "USD",
                "not_actual_invoice": True,
                # A fixture representa uma estimativa completa.
                # Snapshots sem essa confirmação serão tratados
                # conservadoramente como incompletos.
                "is_complete": True,
                "estimated_cost_usd": estimated_cost_usd,
                "known_subtotal_usd": estimated_cost_usd,
                "estimated_cases": 1,
                "incomplete_cases": 0,
            },
        },
    }

    results = []

    if include_case:
        results.append(
            {
                "case_id": "total_orders",
                "question": "Qual o total de pedidos?",
                "status": status,
                "score": case_score,
                "duration_ms": case_duration_ms,
                "cache_hit": cache_hit,
                "provider": provider,
                "model": f"{provider}-model",
                "metadata": {
                    "category": "volume",
                    "llm_metrics": {
                        "provider": provider,
                        "model": f"{provider}-model",
                        "requests": {
                            "total": 1,
                            "successful": 1,
                            "failed": 0,
                            "retries": 0,
                        },
                        "tokens": {
                            "input": input_tokens,
                            "output": output_tokens,
                            "thinking": 0,
                            "total": input_tokens + output_tokens,
                            "unreported": 0,
                        },
                        "latency": {
                            "total_ms": case_duration_ms,
                            "average_ms": case_duration_ms,
                        },
                    },
                },
            }
        )

    return {
        "run_id": run_id,
        "provider": provider,
        "model": f"{provider}-model",
        "summary": {
            "total_cases": 1,
            "evaluated_cases": 1,
            "passed_cases": 1 if status == "passed" else 0,
            "failed_cases": 1 if status == "failed" else 0,
            "error_cases": 1 if status == "error" else 0,
            "skipped_cases": 0,
            "pass_rate": pass_rate,
        },
        "results": results,
        "metadata": {
            "metrics": {
                "summary": summary,
                "by_category": {
                    "volume": summary,
                },
                "by_provider": {
                    provider: summary,
                },
                "by_model": {
                    f"{provider}/{provider}-model": summary,
                },
            },
        },
    }


# ==========================================================
# RUNNER SIMULADO
# ==========================================================


class SequenceRunner:
    """Runner que retorna um relatório diferente por repetição."""

    def __init__(
        self,
        provider: str,
        reports: list[dict],
        created_runners: list[str],
        executions: list[dict],
        failure: Exception | None = None,
    ):
        self.provider = provider
        self.reports = reports
        self.created_runners = created_runners
        self.executions = executions
        self.failure = failure

        self.created_runners.append(provider)

    def run(
        self,
        cases: list[EvaluationCase],
        run_id: str,
    ) -> dict:
        """Executa uma avaliação simulada."""

        self.executions.append(
            {
                "provider": self.provider,
                "run_id": run_id,
                "cases": list(cases),
            }
        )

        if self.failure is not None:
            raise self.failure

        if not self.reports:
            raise RuntimeError("Não existem relatórios simulados disponíveis.")

        report = self.reports.pop(0)
        report["run_id"] = run_id

        return report


# ==========================================================
# FÁBRICA DE BENCHMARK
# ==========================================================


def make_benchmark(
    providers: list[str],
    cases: list[EvaluationCase],
    reports_by_provider: dict[str, list[dict]],
    *,
    repetitions: int,
    failures: dict[tuple[str, int], Exception] | None = None,
):
    """Cria BenchmarkRunner com runners simulados."""

    failures = failures or {}

    executions = []
    created_runners = []

    repetition_counters = defaultdict(int)

    def runner_factory(
        provider_name: str,
        dataset_path=None,
    ):
        repetition_counters[provider_name] += 1

        repetition = repetition_counters[provider_name]

        failure = failures.get(
            (provider_name, repetition),
        )

        return SequenceRunner(
            provider=provider_name,
            reports=reports_by_provider.get(provider_name, []),
            created_runners=created_runners,
            executions=executions,
            failure=failure,
        )

    benchmark = BenchmarkRunner(
        providers=providers,
        runner_factory=runner_factory,
        repetitions=repetitions,
    )

    return benchmark, executions, created_runners


# ==========================================================
# VALIDAÇÃO DE REPETIÇÕES
# ==========================================================


@pytest.mark.parametrize(
    "repetitions",
    [
        0,
        -1,
        -10,
    ],
)
def test_benchmark_rejects_non_positive_repetitions(repetitions):
    """Não permite quantidade de repetições igual ou inferior a zero."""

    with pytest.raises(
        ValueError,
        match="pelo menos 1",
    ):
        BenchmarkRunner(
            providers=["gemini", "openai"],
            repetitions=repetitions,
        )


@pytest.mark.parametrize(
    "repetitions",
    [
        True,
        False,
        1.5,
        "3",
        None,
    ],
)
def test_benchmark_rejects_invalid_repetition_types(repetitions):
    """Rejeita valores que não sejam inteiros."""

    with pytest.raises(
        TypeError,
        match="inteiro",
    ):
        BenchmarkRunner(
            providers=["gemini", "openai"],
            repetitions=repetitions,
        )


def test_benchmark_defaults_to_one_repetition():
    """A execução padrão continua preservando o comportamento anterior."""

    benchmark = BenchmarkRunner(
        providers=["gemini", "openai"],
        runner_factory=lambda **kwargs: None,
    )

    assert benchmark.repetitions == 1


# ==========================================================
# ESTATÍSTICAS DESCRITIVAS
# ==========================================================


def test_summarize_numeric_samples_calculates_all_statistics():
    """Calcula média, mediana, desvio-padrão, mínimo e máximo."""

    values = [1.0, 2.0, 3.0]

    summary = BenchmarkRunner._summarize_numeric_samples(
        values,
        requested_runs=3,
    )

    assert summary["sample_count"] == 3
    assert summary["missing_runs"] == 0

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)

    assert summary["standard_deviation"] == pytest.approx(stdev(values))

    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_summarize_numeric_samples_uses_sample_standard_deviation():
    """Não confunde desvio-padrão amostral com populacional."""

    summary = BenchmarkRunner._summarize_numeric_samples(
        [10.0, 20.0],
        requested_runs=2,
    )

    assert summary["standard_deviation"] == pytest.approx(
        7.071068,
        abs=0.000001,
    )


def test_summarize_numeric_samples_returns_none_for_single_sample_deviation():
    """Uma única observação não possui desvio-padrão amostral."""

    summary = BenchmarkRunner._summarize_numeric_samples(
        [5.0],
        requested_runs=3,
    )

    assert summary["sample_count"] == 1
    assert summary["missing_runs"] == 2

    assert summary["mean"] == 5.0
    assert summary["median"] == 5.0
    assert summary["standard_deviation"] is None

    assert summary["min"] == 5.0
    assert summary["max"] == 5.0


def test_summarize_numeric_samples_handles_empty_values():
    """Métricas totalmente ausentes não recebem valores artificiais."""

    summary = BenchmarkRunner._summarize_numeric_samples(
        [],
        requested_runs=3,
    )

    assert summary["sample_count"] == 0
    assert summary["missing_runs"] == 3

    assert summary["mean"] is None
    assert summary["median"] is None
    assert summary["standard_deviation"] is None
    assert summary["min"] is None
    assert summary["max"] is None


def test_summarize_numeric_samples_ignores_invalid_values():
    """Valores inválidos não contaminam os cálculos."""

    summary = BenchmarkRunner._summarize_numeric_samples(
        [
            10.0,
            True,
            -1.0,
            float("nan"),
            float("inf"),
            "invalid",
            20.0,
        ],
        requested_runs=7,
    )

    assert summary["sample_count"] == 2
    assert summary["missing_runs"] == 5

    assert summary["mean"] == 15.0
    assert summary["median"] == 15.0


# ==========================================================
# REPETIBILIDADE BÁSICA
# ==========================================================


def test_benchmark_executes_requested_repetitions_for_each_provider():
    """Executa três repetições independentes por provedor."""

    cases = [
        make_case(),
    ]

    reports = {
        provider: [make_report(provider, f"run-{index}") for index in range(3)]
        for provider in ("gemini", "openai")
    }

    benchmark, executions, created_runners = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=3,
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="repeatability-basic",
    )

    assert len(executions) == 6
    assert len(created_runners) == 6

    assert report["configuration"]["repetitions_per_provider"] == 3
    assert report["configuration"]["total_requested_runs"] == 6

    for provider in ("gemini", "openai"):
        provider_runs = report["provider_runs"][provider]

        assert provider_runs["execution_status"] == "completed"

        assert len(provider_runs["repetitions"]) == 3

        assert provider_runs["repeatability"]["requested_runs"] == 3
        assert provider_runs["repeatability"]["completed_runs"] == 3
        assert provider_runs["repeatability"]["failed_runs"] == 0

        run_ids = [
            repetition["report"]["run_id"]
            for repetition in provider_runs["repetitions"]
        ]

        assert len(set(run_ids)) == 3

    assert report["summary"]["total_repetitions"] == 6
    assert report["summary"]["completed_repetitions"] == 6
    assert report["summary"]["failed_repetitions"] == 0


def test_benchmark_preserves_each_individual_repetition_report():
    """Todos os relatórios permanecem disponíveis no resultado final."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report(
                "gemini",
                "run-1",
                average_score=0.8,
            ),
            make_report(
                "gemini",
                "run-2",
                average_score=0.6,
            ),
        ],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=2,
    )

    report = benchmark.run(cases=cases)

    gemini_runs = report["provider_runs"]["gemini"]["repetitions"]

    first_report = gemini_runs[0]["report"]
    second_report = gemini_runs[1]["report"]

    assert first_report["metadata"]["metrics"]["summary"]["average_score"] == 0.8

    assert second_report["metadata"]["metrics"]["summary"]["average_score"] == 0.6

    assert all(
        repetition["execution_status"] == "completed" for repetition in gemini_runs
    )


# ==========================================================
# ESTATÍSTICAS POR PROVEDOR
# ==========================================================


def test_benchmark_calculates_provider_repeatability_statistics():
    """Consolida estatísticas das métricas de cada provedor."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report(
                "gemini",
                "run-1",
                pass_rate=100.0,
                average_score=0.9,
                duration_ms=100.0,
                cache_hit_rate=0.0,
                estimated_cost_usd=0.001,
                input_tokens=100,
                output_tokens=50,
            ),
            make_report(
                "gemini",
                "run-2",
                pass_rate=50.0,
                average_score=0.6,
                duration_ms=200.0,
                cache_hit_rate=50.0,
                estimated_cost_usd=0.002,
                input_tokens=200,
                output_tokens=100,
            ),
            make_report(
                "gemini",
                "run-3",
                pass_rate=0.0,
                average_score=0.3,
                duration_ms=300.0,
                cache_hit_rate=100.0,
                estimated_cost_usd=0.003,
                input_tokens=300,
                output_tokens=150,
            ),
        ],
        "openai": [
            make_report(
                "openai",
                "run-1",
                pass_rate=100.0,
                average_score=0.8,
                duration_ms=120.0,
            ),
            make_report(
                "openai",
                "run-2",
                pass_rate=100.0,
                average_score=0.8,
                duration_ms=120.0,
            ),
            make_report(
                "openai",
                "run-3",
                pass_rate=100.0,
                average_score=0.8,
                duration_ms=120.0,
            ),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=3,
    )

    report = benchmark.run(cases=cases)

    repeatability = report["repeatability"]

    gemini = repeatability["by_provider"]["gemini"]

    assert gemini["requested_runs"] == 3
    assert gemini["completed_runs"] == 3
    assert gemini["failed_runs"] == 0
    assert gemini["status"] == "completed"

    # Taxa de aprovação.

    pass_rate = gemini["metrics"]["pass_rate"]

    assert pass_rate["mean"] == 50.0
    assert pass_rate["median"] == 50.0
    assert pass_rate["standard_deviation"] == 50.0
    assert pass_rate["min"] == 0.0
    assert pass_rate["max"] == 100.0

    # Score.

    score = gemini["metrics"]["average_score"]

    assert score["mean"] == 0.6
    assert score["median"] == 0.6
    assert score["standard_deviation"] == 0.3

    # Duração.

    duration = gemini["metrics"]["average_duration_ms"]

    assert duration["mean"] == 200.0
    assert duration["median"] == 200.0
    assert duration["standard_deviation"] == 100.0
    assert duration["min"] == 100.0
    assert duration["max"] == 300.0

    # Cache.

    cache = gemini["metrics"]["cache_hit_rate"]

    assert cache["mean"] == 50.0
    assert cache["median"] == 50.0
    assert cache["min"] == 0.0
    assert cache["max"] == 100.0

    # Custo.

    cost = gemini["metrics"]["estimated_cost_usd"]

    assert cost["mean"] == 0.002
    assert cost["median"] == 0.002
    assert cost["standard_deviation"] == 0.001

    # Tokens.

    tokens = gemini["metrics"]["input_tokens"]

    assert tokens["mean"] == 200.0
    assert tokens["median"] == 200.0
    assert tokens["standard_deviation"] == 100.0

    # OpenAI mantém estabilidade nas métricas simuladas.

    openai = repeatability["by_provider"]["openai"]

    assert openai["metrics"]["pass_rate"]["mean"] == 100.0
    assert openai["metrics"]["pass_rate"]["standard_deviation"] == 0.0

    assert openai["metrics"]["average_score"]["mean"] == 0.8
    assert openai["metrics"]["average_score"]["standard_deviation"] == 0.0

    assert openai["metrics"]["average_duration_ms"]["mean"] == 120.0


# ==========================================================
# REPETIBILIDADE POR CENÁRIO
# ==========================================================


def test_benchmark_calculates_repeatability_by_case():
    """Analisa a estabilidade de cada cenário entre execuções."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report(
                "gemini",
                "run-1",
                status="passed",
                case_score=1.0,
                case_duration_ms=100.0,
                cache_hit=False,
            ),
            make_report(
                "gemini",
                "run-2",
                status="failed",
                case_score=0.5,
                case_duration_ms=200.0,
                cache_hit=True,
            ),
            make_report(
                "gemini",
                "run-3",
                status="error",
                case_score=0.2,
                case_duration_ms=300.0,
                cache_hit=True,
            ),
        ],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
            make_report("openai", "run-3"),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=3,
    )

    report = benchmark.run(cases=cases)

    gemini = report["repeatability"]["by_provider"]["gemini"]

    case = gemini["by_case"]["total_orders"]

    assert case["requested_runs"] == 3
    assert case["observed_runs"] == 3
    assert case["missing_runs"] == 0

    assert case["passed_runs"] == 1
    assert case["failed_or_error_runs"] == 2

    pass_rate = case["metrics"]["pass_rate"]

    assert pass_rate["sample_count"] == 3
    assert pass_rate["mean"] == pytest.approx(
        33.333333,
        abs=0.000001,
    )

    score = case["metrics"]["score"]

    assert score["mean"] == pytest.approx(
        0.566667,
        abs=0.000001,
    )

    duration = case["metrics"]["duration_ms"]

    assert duration["mean"] == 200.0
    assert duration["min"] == 100.0
    assert duration["max"] == 300.0


def test_benchmark_marks_missing_case_results():
    """Resultados ausentes não são tratados como aprovações."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report("gemini", "run-1"),
            make_report(
                "gemini",
                "run-2",
                include_case=False,
            ),
            make_report("gemini", "run-3"),
        ],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
            make_report("openai", "run-3"),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=3,
    )

    report = benchmark.run(cases=cases)

    case = report["repeatability"]["by_provider"]["gemini"]["by_case"]["total_orders"]

    assert case["requested_runs"] == 3
    assert case["observed_runs"] == 2
    assert case["missing_runs"] == 1

    assert case["metrics"]["score"]["sample_count"] == 2
    assert case["metrics"]["score"]["missing_runs"] == 1


# ==========================================================
# ISOLAMENTO DE FALHAS
# ==========================================================


def test_benchmark_continues_after_individual_repetition_failure():
    """Uma repetição com falha não interrompe as seguintes."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report("gemini", "run-2"),
            make_report("gemini", "run-3"),
        ],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
            make_report("openai", "run-3"),
        ],
    }

    benchmark, executions, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=3,
        failures={
            ("gemini", 1): RuntimeError("Falha simulada na primeira repetição."),
        },
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="failure-isolation-repeatability",
    )

    gemini = report["provider_runs"]["gemini"]

    # Todas as tentativas são registradas, inclusive as que falham.
    assert len(executions) == 6

    assert gemini["execution_status"] == "completed"

    assert gemini["repeatability"]["requested_runs"] == 3
    assert gemini["repeatability"]["completed_runs"] == 2
    assert gemini["repeatability"]["failed_runs"] == 1

    assert gemini["repeatability"]["status"] == "completed_with_failures"

    statuses = [run["execution_status"] for run in gemini["repetitions"]]

    assert statuses == [
        "error",
        "completed",
        "completed",
    ]

    assert gemini["repetitions"][0]["error"] == (
        "Falha simulada na primeira repetição."
    )

    assert gemini["report"]["run_id"] == (
        "failure-isolation-repeatability-gemini-rep-002"
    )

    assert report["summary"]["failed_repetitions"] == 1
    assert report["summary"]["completed_repetitions"] == 5


def test_benchmark_reports_provider_as_failed_when_all_repetitions_fail():
    """Um provedor sem execuções bem-sucedidas é marcado como erro."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=2,
        failures={
            ("gemini", 1): RuntimeError("Erro 1"),
            ("gemini", 2): RuntimeError("Erro 2"),
        },
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="all-failures",
    )

    gemini = report["provider_runs"]["gemini"]

    assert gemini["execution_status"] == "error"
    assert gemini["report"] is None

    assert gemini["repeatability"]["completed_runs"] == 0
    assert gemini["repeatability"]["failed_runs"] == 2
    assert gemini["repeatability"]["status"] == "failed"

    assert report["comparison"]["by_provider"]["gemini"]["metrics"] is None

    assert (
        report["comparison"]["by_provider"]["gemini"]["repeatability"]["failed_runs"]
        == 2
    )


def test_benchmark_uses_first_successful_repetition_as_representative():
    """A comparação tradicional utiliza o primeiro relatório bem-sucedido."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report(
                "gemini",
                "run-2",
                average_score=0.75,
            ),
            make_report(
                "gemini",
                "run-3",
                average_score=0.50,
            ),
        ],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
            make_report("openai", "run-3"),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=3,
        failures={
            ("gemini", 1): RuntimeError("Primeira execução indisponível."),
        },
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="representative-test",
    )

    comparison = report["comparison"]["by_provider"]["gemini"]

    assert comparison["execution_status"] == "completed"

    assert comparison["run_id"] == ("representative-test-gemini-rep-002")

    assert comparison["metrics"]["average_score"] == 0.75

    assert (
        report["provider_runs"]["gemini"]["repeatability"]["representative_run_id"]
        == "representative-test-gemini-rep-002"
    )


# ==========================================================
# COMPARABILIDADE E COMPATIBILIDADE
# ==========================================================


def test_default_benchmark_does_not_add_repeatability_structure():
    """Uma repetição mantém o contrato original dos relatórios."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report("gemini", "run-gemini"),
        ],
        "openai": [
            make_report("openai", "run-openai"),
        ],
    }

    benchmark, executions, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=1,
    )

    report = benchmark.run(
        cases=cases,
        benchmark_id="compatibility-test",
    )

    assert len(executions) == 2

    assert "repeatability" not in report

    assert "repetitions" not in report["provider_runs"]["gemini"]

    assert report["provider_runs"]["gemini"]["report"]["run_id"] == (
        "compatibility-test-gemini"
    )


def test_repetition_reports_keep_provider_isolation():
    """As execuções de provedores diferentes não se misturam."""

    cases = [
        make_case(),
    ]

    reports = {
        "gemini": [
            make_report("gemini", "run-1"),
            make_report("gemini", "run-2"),
        ],
        "openai": [
            make_report("openai", "run-1"),
            make_report("openai", "run-2"),
        ],
    }

    benchmark, _, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=2,
    )

    report = benchmark.run(cases=cases)

    gemini_runs = report["provider_runs"]["gemini"]["repetitions"]
    openai_runs = report["provider_runs"]["openai"]["repetitions"]

    assert all(run["report"]["provider"] == "gemini" for run in gemini_runs)

    assert all(run["report"]["provider"] == "openai" for run in openai_runs)

    gemini_ids = {run["report"]["run_id"] for run in gemini_runs}

    openai_ids = {run["report"]["run_id"] for run in openai_runs}

    assert gemini_ids.isdisjoint(openai_ids)


def test_repetition_reports_keep_same_dataset():
    """Todas as repetições recebem exatamente os mesmos cenários."""

    cases = [
        make_case(),
        make_case(
            case_id="delivery",
            category="delivery",
        ),
    ]

    reports = {
        provider: [
            make_report(provider, "run-1"),
            make_report(provider, "run-2"),
        ]
        for provider in ("gemini", "openai")
    }

    benchmark, executions, _ = make_benchmark(
        providers=["gemini", "openai"],
        cases=cases,
        reports_by_provider=reports,
        repetitions=2,
    )

    benchmark.run(cases=cases)

    assert len(executions) == 4

    for execution in executions:
        assert execution["cases"] == cases


# ==========================================================
# REGRESSÃO DE COMPLETUDE FINANCEIRA
# ==========================================================


def test_legacy_cost_without_completeness_flag_is_not_total():
    """Snapshots antigos sem confirmação não entram no custo total."""

    report = make_report(
        provider="gemini",
        run_id="legacy-cost",
        estimated_cost_usd=0.002,
    )

    cost_estimate = report["metadata"]["metrics"]["summary"]["llm_usage"][
        "token_cost_estimate"
    ]

    # Simula um snapshot legado sem confirmação de completude.
    cost_estimate.pop("is_complete")

    summary = BenchmarkRunner._build_repeatability_summary(
        repetition_runs=[
            {
                "execution_status": "completed",
                "report": report,
            },
        ],
        requested_runs=1,
    )

    estimated_cost = summary["metrics"]["estimated_cost_usd"]
    known_subtotal = summary["metrics"]["known_cost_subtotal_usd"]

    assert estimated_cost["sample_count"] == 0
    assert estimated_cost["mean"] is None

    assert known_subtotal["sample_count"] == 1
    assert known_subtotal["mean"] == pytest.approx(0.002)

    assert summary["cost_observations"]["complete_runs"] == 0
    assert summary["cost_observations"]["incomplete_runs"] == 1
