from __future__ import annotations

from typing import Any

import pytest

from ai_analyst.evaluation.models import EvaluationCase
from ai_analyst.evaluation.runner import EvaluationRunner
from ai_analyst.llm.metrics import LLMMetrics

# ==========================================================
# DEPENDÊNCIAS SIMULADAS
# ==========================================================


class FakeReferenceExecutor:
    """Simula a execução dos SQLs de referência."""

    def __init__(
        self,
        responses: dict[str, Any],
    ):
        self.responses = responses
        self.executed_queries: list[str] = []

    def execute(self, sql: str) -> list[dict[str, Any]]:
        self.executed_queries.append(sql)

        response = self.responses[sql]

        if isinstance(response, Exception):
            raise response

        return [dict(row) for row in response]


class FakeAnalyst:
    """Simula o BusinessAnalyst sem utilizar APIs reais."""

    model = "fake-model"

    def __init__(
        self,
        responses: list[Any],
    ):
        self.responses = list(responses)
        self.questions: list[str] = []

    def analyze(self, question: str) -> dict[str, Any]:
        self.questions.append(question)

        if not self.responses:
            raise RuntimeError("Não existem respostas simuladas disponíveis.")

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


# ==========================================================
# FÁBRICAS DE OBJETOS DE TESTE
# ==========================================================


def make_case(
    case_id: str = "test_case",
    question: str = "Quantos pedidos existem?",
    reference_sql: str = "SELECT 1",
    category: str = "volume",
    required_columns: tuple[str, ...] = (),
    expected_analysis_mode: str | None = None,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=question,
        reference_sql=reference_sql,
        category=category,
        comparison_mode="exact",
        required_columns=required_columns,
        expected_analysis_mode=expected_analysis_mode,
    )


def make_analysis(
    results: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    analysis = {
        "analysis_id": "analysis-test-001",
        "provider": "gemini",
        "model": "fake-model",
        "analysis_mode": "single_query",
        "results": results or [],
        "total_results": len(results or []),
        "cache_hit": False,
        "steps": [],
    }

    analysis.update(overrides)

    return analysis


def make_runner(
    analyst: FakeAnalyst,
    reference_responses: dict[str, Any],
    provider_name: str = "gemini",
) -> EvaluationRunner:
    return EvaluationRunner(
        provider_name=provider_name,
        analyst_factory=lambda **kwargs: analyst,
        reference_executor=FakeReferenceExecutor(reference_responses),
    )


# ==========================================================
# INICIALIZAÇÃO
# ==========================================================


def test_runner_uses_gemini_as_default_provider():
    analyst = FakeAnalyst([])

    runner = make_runner(
        analyst=analyst,
        reference_responses={},
    )

    assert runner.provider_name == "gemini"


@pytest.mark.parametrize(
    "provider",
    ["gemini", "openai", "anthropic"],
)
def test_runner_accepts_supported_providers(provider):
    analyst = FakeAnalyst([])

    runner = make_runner(
        analyst=analyst,
        reference_responses={},
        provider_name=provider,
    )

    assert runner.provider_name == provider


def test_runner_normalizes_provider_name():
    runner = EvaluationRunner(
        provider_name=" OPENAI ",
        analyst_factory=lambda **kwargs: None,
        reference_executor=FakeReferenceExecutor({}),
    )

    assert runner.provider_name == "openai"


def test_runner_rejects_unsupported_provider():
    with pytest.raises(
        ValueError,
        match="Provedor não suportado",
    ):
        EvaluationRunner(
            provider_name="invalid-provider",
        )


# ==========================================================
# EXECUÇÃO DE CENÁRIOS SIMPLES
# ==========================================================


def test_runner_executes_successful_scenario():
    case = make_case()

    analyst = FakeAnalyst(
        [
            make_analysis(
                [
                    {"total_orders": 60},
                ]
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(
        cases=[case],
        run_id="test-run-001",
    )

    assert report.run_id == "test-run-001"

    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.failed_cases == 0
    assert report.error_cases == 0

    assert report.pass_rate == 100.0

    assert report.finished_at is not None

    result = report.results[0]

    assert result.status == "passed"
    assert result.score == 1.0

    assert result.provider == "gemini"
    assert result.model == "fake-model"

    assert result.analysis_id == "analysis-test-001"

    # Compatibilidade com cenários sem expectativa de modo.
    assert case.expected_analysis_mode is None
    assert "expected_analysis_mode" not in result.metadata
    assert "analysis_mode_matches" not in result.metadata
    assert "analysis_mode_mismatch" not in result.metadata

    assert analyst.questions == [
        "Quantos pedidos existem?",
    ]


def test_runner_records_failed_comparison():
    case = make_case()

    analyst = FakeAnalyst(
        [
            make_analysis(
                [
                    {"total_orders": 50},
                ]
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=[case])

    assert report.total_cases == 1
    assert report.failed_cases == 1
    assert report.passed_cases == 0

    result = report.results[0]

    assert result.status == "failed"
    assert result.score < 1.0
    assert len(result.differences) > 0


def test_runner_records_cache_hit():
    case = make_case()

    analyst = FakeAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
                cache_hit=True,
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "passed"
    assert result.cache_hit is True

    summary = report.metadata["metrics"]["summary"]

    assert summary["cache_observations"] == 1
    assert summary["cache_hits"] == 1
    assert summary["cache_misses"] == 0
    assert summary["cache_hit_rate"] == 100.0


# ==========================================================
# TRATAMENTO DE ERROS
# ==========================================================


def test_runner_continues_after_reference_sql_failure():
    first_case = make_case(
        case_id="reference_error",
        reference_sql="INVALID SQL",
    )

    second_case = make_case(
        case_id="successful_case",
        reference_sql="SELECT 2",
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                [
                    {"total_orders": 60},
                ]
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "INVALID SQL": RuntimeError("Falha simulada no SQL de referência."),
            "SELECT 2": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(
        cases=[
            first_case,
            second_case,
        ],
    )

    assert report.total_cases == 2

    assert report.error_cases == 1
    assert report.passed_cases == 1

    assert report.results[0].status == "error"
    assert report.results[0].metadata["phase"] == "reference_sql"

    assert report.results[1].status == "passed"

    assert len(analyst.questions) == 1


def test_runner_continues_after_agent_execution_failure():
    first_case = make_case(
        case_id="agent_error",
        reference_sql="SELECT 1",
    )

    second_case = make_case(
        case_id="successful_case",
        reference_sql="SELECT 2",
    )

    analyst = FakeAnalyst(
        [
            RuntimeError("Falha simulada no agente."),
            make_analysis(
                [
                    {"total_orders": 60},
                ]
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 10},
            ],
            "SELECT 2": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(
        cases=[
            first_case,
            second_case,
        ],
    )

    assert report.total_cases == 2

    assert report.error_cases == 1
    assert report.passed_cases == 1

    assert report.results[0].status == "error"

    assert report.results[0].metadata["phase"] == "agent_execution"

    assert report.results[1].status == "passed"


def test_runner_handles_empty_dataset():
    analyst_factory_called = False

    def analyst_factory(**kwargs):
        nonlocal analyst_factory_called

        analyst_factory_called = True

        return FakeAnalyst([])

    runner = EvaluationRunner(
        analyst_factory=analyst_factory,
        reference_executor=FakeReferenceExecutor({}),
    )

    report = runner.run(cases=[])

    assert report.total_cases == 0
    assert report.passed_cases == 0
    assert report.pass_rate == 0.0

    assert report.finished_at is not None

    assert analyst_factory_called is False

    summary = report.metadata["metrics"]["summary"]

    assert summary["total_cases"] == 0
    assert summary["average_score"] == 0.0


def test_runner_rejects_invalid_case_objects():
    analyst = FakeAnalyst([])

    runner = make_runner(
        analyst=analyst,
        reference_responses={},
    )

    with pytest.raises(
        TypeError,
        match="EvaluationCase",
    ):
        runner.run(cases=["invalid-case"])


# ==========================================================
# ANÁLISES COMPOSTAS
# ==========================================================


def test_runner_evaluates_multi_query_with_complete_result():
    case = make_case(
        required_columns=("city", "order_count"),
        expected_analysis_mode="multi_query",
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[],
                analysis_mode="multi_query",
                queries=[
                    {
                        "id": "query-1",
                        "status": "success",
                        "results": [
                            {
                                "city": "São Paulo",
                                "order_count": 100,
                            },
                            {
                                "city": "Curitiba",
                                "order_count": 50,
                            },
                        ],
                    },
                    {
                        "id": "query-2",
                        "status": "success",
                        "results": [
                            {
                                "other_metric": 999,
                            },
                        ],
                    },
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {
                    "city": "São Paulo",
                    "order_count": 100,
                },
                {
                    "city": "Curitiba",
                    "order_count": 50,
                },
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "passed"

    assert result.metadata["analysis_mode"] == "multi_query"
    assert result.metadata["expected_analysis_mode"] == "multi_query"
    assert result.metadata["analysis_mode_matches"] is True

    assert "analysis_mode_mismatch" not in result.metadata

    assert result.metadata["selected_query_ids"] == [
        "query-1",
    ]


def test_runner_combines_compatible_scalar_results():
    case = make_case(
        required_columns=(
            "total_orders",
            "cancelled_orders",
        ),
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[],
                analysis_mode="multi_query",
                queries=[
                    {
                        "id": "query-1",
                        "status": "success",
                        "results": [
                            {"total_orders": 60_000},
                        ],
                    },
                    {
                        "id": "query-2",
                        "status": "success",
                        "results": [
                            {"cancelled_orders": 2_172},
                        ],
                    },
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {
                    "total_orders": 60_000,
                    "cancelled_orders": 2_172,
                },
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "passed"

    assert result.actual_rows == [
        {
            "total_orders": 60_000,
            "cancelled_orders": 2_172,
        },
    ]

    assert result.metadata["results_combined"] is True

    assert result.metadata["selected_query_ids"] == [
        "query-1",
        "query-2",
    ]


def test_runner_rejects_conflicting_scalar_results():
    case = make_case(
        required_columns=(
            "total_orders",
            "cancelled_orders",
        ),
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[],
                analysis_mode="multi_query",
                queries=[
                    {
                        "id": "query-1",
                        "status": "success",
                        "results": [
                            {
                                "total_orders": 60_000,
                            },
                        ],
                    },
                    {
                        "id": "query-2",
                        "status": "success",
                        "results": [
                            {
                                "total_orders": 59_000,
                                "cancelled_orders": 2_172,
                            },
                        ],
                    },
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {
                    "total_orders": 60_000,
                    "cancelled_orders": 2_172,
                },
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "skipped"

    assert result.error == "multi_query_results_not_combinable"


def test_runner_does_not_join_tables_using_unique_columns():
    case = make_case(
        required_columns=(
            "city",
            "order_count",
            "revenue",
        ),
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[],
                analysis_mode="multi_query",
                queries=[
                    {
                        "id": "query-1",
                        "status": "success",
                        "results": [
                            {
                                "city": "São Paulo",
                                "order_count": 100,
                            },
                            {
                                "city": "Curitiba",
                                "order_count": 50,
                            },
                        ],
                    },
                    {
                        "id": "query-2",
                        "status": "success",
                        "results": [
                            {
                                "city": "São Paulo",
                                "revenue": 1000,
                            },
                            {
                                "city": "Curitiba",
                                "revenue": 500,
                            },
                        ],
                    },
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {
                    "city": "São Paulo",
                    "order_count": 100,
                    "revenue": 1000,
                },
                {
                    "city": "Curitiba",
                    "order_count": 50,
                    "revenue": 500,
                },
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "skipped"

    assert result.error == "multi_query_results_not_combinable"


def test_runner_marks_unsupported_capability_as_skipped():
    case = make_case()

    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[],
                analysis_mode="multi_query",
                queries=[
                    {
                        "id": "query-1",
                        "status": "failure",
                        "failure_type": "unsupported_capability",
                        "results": [],
                    },
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "skipped"

    assert result.error == "unsupported_capability"

    assert result.metadata["unsupported_queries"] == 1


# ==========================================================
# EXPECTATIVA DO MODO DE ANÁLISE
# ==========================================================


def test_runner_fails_when_single_query_is_used_instead_of_expected_multi_query():
    case = make_case(
        expected_analysis_mode="multi_query",
    )

    # O SQL retornado pelo agente está correto.
    # A única divergência é o modo de execução.
    analyst = FakeAnalyst(
        [
            make_analysis(
                [
                    {"total_orders": 60},
                ]
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.error_cases == 0

    assert report.pass_rate == 0.0

    assert result.status == "failed"

    assert result.score == 1.0
    assert result.metadata["result_comparison_status"] == "passed"
    assert result.metadata["result_comparison_score"] == 1.0

    assert result.metadata["analysis_mode"] == "single_query"
    assert result.metadata["expected_analysis_mode"] == "multi_query"
    assert result.metadata["analysis_mode_matches"] is False

    assert result.metadata["analysis_mode_mismatch"] == {
        "expected": "multi_query",
        "actual": "single_query",
    }


def test_runner_fails_when_multi_query_is_used_instead_of_expected_single_query():
    case = make_case(
        expected_analysis_mode="single_query",
    )

    # O resultado combinado continua correto,
    # mas o agente utilizou multi_query indevidamente.
    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[],
                analysis_mode="multi_query",
                queries=[
                    {
                        "id": "query-1",
                        "status": "success",
                        "results": [
                            {"total_orders": 60},
                        ],
                    },
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "failed"
    assert result.score == 1.0

    assert result.metadata["analysis_mode"] == "multi_query"
    assert result.metadata["expected_analysis_mode"] == "single_query"
    assert result.metadata["analysis_mode_matches"] is False

    assert result.metadata["analysis_mode_mismatch"] == {
        "expected": "single_query",
        "actual": "multi_query",
    }


def test_runner_preserves_operational_error_when_analysis_mode_mismatches():
    case = make_case(
        expected_analysis_mode="multi_query",
    )

    # O agente informa single_query, mas devolve resultados
    # em formato inválido. A falha operacional deve prevalecer.
    analyst = FakeAnalyst(
        [
            make_analysis(
                results="invalid-results",
                analysis_mode="single_query",
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "error"

    assert report.error_cases == 1
    assert report.failed_cases == 0

    assert result.metadata["phase"] == "result_extraction"

    assert result.metadata["analysis_mode"] == "single_query"
    assert result.metadata["expected_analysis_mode"] == "multi_query"
    assert result.metadata["analysis_mode_matches"] is False

    assert result.metadata["analysis_mode_mismatch"] == {
        "expected": "multi_query",
        "actual": "single_query",
    }

    assert "result_comparison_status" not in result.metadata


# ==========================================================
# MÉTRICAS E RELATÓRIO
# ==========================================================


def test_runner_generates_consolidated_metrics():
    cases = [
        make_case(
            case_id="case-1",
            reference_sql="SELECT 1",
            category="volume",
        ),
        make_case(
            case_id="case-2",
            reference_sql="SELECT 2",
            category="financial",
        ),
    ]

    analyst = FakeAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
                cache_hit=False,
            ),
            make_analysis(
                [{"total_orders": 60}],
                cache_hit=True,
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
            "SELECT 2": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(cases=cases)

    metrics = report.metadata["metrics"]

    summary = metrics["summary"]

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 2
    assert summary["failed_cases"] == 0

    assert summary["pass_rate"] == 100.0

    assert summary["cache_observations"] == 2
    assert summary["cache_hits"] == 1
    assert summary["cache_misses"] == 1
    assert summary["cache_hit_rate"] == 50.0

    assert "volume" in metrics["by_category"]
    assert "financial" in metrics["by_category"]

    assert "gemini" in metrics["by_provider"]

    assert "gemini/fake-model" in metrics["by_model"]


def test_report_serialization_includes_metrics():
    case = make_case()

    analyst = FakeAnalyst(
        [
            make_analysis(
                [
                    {"total_orders": 60},
                ]
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 60},
            ],
        },
    )

    report = runner.run(
        cases=[case],
        run_id="serialization-test",
    )

    serialized = report.to_dict()

    assert serialized["run_id"] == "serialization-test"

    assert serialized["finished_at"] is not None

    assert serialized["summary"]["passed_cases"] == 1

    assert "metrics" in serialized["metadata"]

    assert serialized["metadata"]["metrics"]["summary"]["pass_rate"] == 100.0


# ==========================================================
# INTEGRAÇÃO DAS MÉTRICAS INDIVIDUAIS DE LLM
# ==========================================================


def test_runner_captures_llm_usage_for_successful_scenario(monkeypatch):
    """Captura tokens, requisições, retries e latência do cenário."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class MetricsAnalyst(FakeAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            metrics.record_success(
                model="fake-model",
                duration_ms=120.0,
                prompt_tokens=100,
                output_tokens=50,
                thinking_tokens=10,
                total_tokens=160,
                provider="gemini",
            )

            metrics.record_retry(provider="gemini")

            metrics.record_success(
                model="fake-model",
                duration_ms=80.0,
                prompt_tokens=30,
                output_tokens=20,
                thinking_tokens=5,
                total_tokens=55,
                provider="gemini",
            )

            return super().analyze(question)

    case = make_case()

    analyst = MetricsAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
    )

    report = runner.run(
        cases=[case],
        run_id="llm-usage-success",
    )

    assert report.passed_cases == 1

    result = report.results[0]

    assert result.status == "passed"

    usage = result.metadata["llm_metrics"]

    assert usage["analysis_id"]
    assert usage["provider"] == "gemini"
    assert usage["model"] == "fake-model"

    assert usage["requests"] == {
        "total": 2,
        "successful": 2,
        "failed": 0,
        "retries": 1,
    }

    assert usage["tokens"] == {
        "input": 130,
        "output": 70,
        "thinking": 15,
        "total": 215,
        "unreported": 0,
    }

    assert usage["latency"] == {
        "total_ms": 200.0,
        "average_ms": 100.0,
    }


def test_runner_captures_llm_usage_when_provider_fails(monkeypatch):
    """Preserva as métricas de chamadas que falharam."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class FailingMetricsAnalyst:
        model = "fake-model"

        def analyze(self, question: str) -> dict[str, Any]:
            metrics.record_failure(
                model="fake-model",
                duration_ms=450.0,
                provider="openai",
            )

            raise RuntimeError("Falha simulada no provedor.")

    case = make_case()

    runner = EvaluationRunner(
        provider_name="openai",
        analyst_factory=lambda **kwargs: FailingMetricsAnalyst(),
        reference_executor=FakeReferenceExecutor(
            {
                "SELECT 1": [{"total_orders": 60}],
            }
        ),
    )

    report = runner.run(
        cases=[case],
        run_id="llm-usage-failure",
    )

    assert report.error_cases == 1

    result = report.results[0]

    assert result.status == "error"

    usage = result.metadata["llm_metrics"]

    assert usage["provider"] == "openai"
    assert usage["model"] == "fake-model"

    assert usage["requests"] == {
        "total": 1,
        "successful": 0,
        "failed": 1,
        "retries": 0,
    }

    assert usage["tokens"] == {
        "input": 0,
        "output": 0,
        "thinking": 0,
        "total": 0,
        "unreported": 1,
    }

    assert usage["latency"] == {
        "total_ms": 450.0,
        "average_ms": 450.0,
    }


def test_runner_isolates_llm_metrics_between_scenarios(monkeypatch):
    """Uma execução não deve herdar o consumo de outra."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class PerScenarioMetricsAnalyst(FakeAnalyst):
        def __init__(self, responses):
            super().__init__(responses)
            self.call_count = 0

        def analyze(self, question: str) -> dict[str, Any]:
            self.call_count += 1

            metrics.record_success(
                model="fake-model",
                duration_ms=100.0,
                prompt_tokens=self.call_count * 10,
                output_tokens=self.call_count * 3,
                provider="gemini",
            )

            return super().analyze(question)

    cases = [
        make_case(
            case_id="scenario-1",
            reference_sql="SELECT 1",
        ),
        make_case(
            case_id="scenario-2",
            reference_sql="SELECT 2",
        ),
    ]

    analyst = PerScenarioMetricsAnalyst(
        [
            make_analysis([{"total_orders": 60}]),
            make_analysis([{"total_orders": 80}]),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
            "SELECT 2": [{"total_orders": 80}],
        },
    )

    report = runner.run(
        cases=cases,
        run_id="llm-usage-isolation",
    )

    assert report.passed_cases == 2

    first_usage = report.results[0].metadata["llm_metrics"]
    second_usage = report.results[1].metadata["llm_metrics"]

    assert first_usage["requests"]["total"] == 1
    assert second_usage["requests"]["total"] == 1

    assert first_usage["tokens"]["input"] == 10
    assert second_usage["tokens"]["input"] == 20

    assert first_usage["tokens"]["output"] == 3
    assert second_usage["tokens"]["output"] == 6

    assert first_usage["tokens"]["total"] == 13
    assert second_usage["tokens"]["total"] == 26

    assert first_usage["latency"]["total_ms"] == 100.0
    assert second_usage["latency"]["total_ms"] == 100.0
