from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from ai_analyst.evaluation.evaluator import ResultEvaluator
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


@pytest.mark.parametrize(
    "provider",
    [
        None,
        123,
        ["gemini"],
        {"provider": "gemini"},
    ],
)
def test_runner_rejects_non_string_provider(provider):
    """O provedor precisa ser informado como uma string."""

    with pytest.raises(TypeError):
        EvaluationRunner(
            provider_name=provider,
        )


def test_runner_rejects_empty_provider():
    """Uma string vazia não pode ser normalizada para um provedor válido."""

    with pytest.raises(ValueError):
        EvaluationRunner(
            provider_name="   ",
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

    assert case.expected_analysis_mode is None
    assert "expected_analysis_mode" not in result.metadata
    assert "analysis_mode_matches" not in result.metadata
    assert "analysis_mode_mismatch" not in result.metadata

    # Análises legadas sem evidências continuam compatíveis.
    assert result.metadata["answer_evidence_status"] == "not_evaluated"

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


def test_runner_continues_after_unexpected_case_exception(monkeypatch):
    """Uma exceção inesperada em um cenário não interrompe o benchmark."""

    first_case = make_case(
        case_id="unexpected_error",
        reference_sql="SELECT 1",
    )

    second_case = make_case(
        case_id="successful_after_unexpected_error",
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
            "SELECT 1": [{"total_orders": 60}],
            "SELECT 2": [{"total_orders": 60}],
        },
    )

    original_run_case = runner._run_case

    def run_case_with_unexpected_failure(*args, **kwargs):
        case = kwargs.get("case")

        if case is None and args:
            case = args[0]

        if case.case_id == "unexpected_error":
            raise RuntimeError("Falha inesperada simulada no cenário.")

        return original_run_case(*args, **kwargs)

    monkeypatch.setattr(
        runner,
        "_run_case",
        run_case_with_unexpected_failure,
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
    assert report.results[1].status == "passed"

    assert analyst.questions == [
        second_case.question,
    ]


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


def test_runner_does_not_fallback_to_root_results_for_empty_multi_query():
    """Uma análise multi_query não pode usar results como fallback."""

    case = make_case(
        required_columns=("total_orders",),
        expected_analysis_mode="multi_query",
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                results=[
                    {"total_orders": 60},
                ],
                analysis_mode="multi_query",
                queries=[],
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

    # O resultado raiz coincide com o SQL de referência, mas não
    # pode ser aproveitado quando não existem consultas individuais.
    assert result.status != "passed"
    assert result.actual_rows != [{"total_orders": 60}]


def test_runner_does_not_fallback_when_multi_query_list_is_missing():
    """A ausência de queries também não autoriza fallback para results."""

    case = make_case(
        required_columns=("total_orders",),
        expected_analysis_mode="multi_query",
    )

    analysis = make_analysis(
        results=[
            {"total_orders": 60},
        ],
        analysis_mode="multi_query",
    )

    analysis.pop("queries", None)

    analyst = FakeAnalyst([analysis])

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

    assert result.status != "passed"
    assert result.actual_rows != [{"total_orders": 60}]


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


def test_runner_rejects_unsupported_expected_analysis_mode():
    """O runner deve rejeitar modos inválidos antes de executar o cenário."""

    case = make_case()

    # Simula um objeto inconsistente que ultrapassou a validação
    # original do EvaluationCase.
    object.__setattr__(
        case,
        "expected_analysis_mode",
        "unsupported_mode",
    )

    analyst = FakeAnalyst([])

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
    )

    with pytest.raises(ValueError):
        runner.run(cases=[case])

    # A validação deve ocorrer antes de chamar o agente.
    assert analyst.questions == []


def test_runner_rejects_unsupported_actual_analysis_mode():
    """Um modo retornado pelo agente precisa pertencer ao contrato suportado."""

    case = make_case(
        expected_analysis_mode="single_query",
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
                analysis_mode="unsupported_mode",
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "error"
    assert result.error


def test_runner_preserves_operational_error_when_analysis_mode_mismatches():
    case = make_case(
        expected_analysis_mode="multi_query",
    )

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
# IDENTIDADE E METADADOS DA ANÁLISE
# ==========================================================


def test_runner_rejects_unsupported_provider_returned_by_analysis():
    """O agente não pode substituir o provedor por um valor desconhecido."""

    analyst = FakeAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
                provider="unsupported-provider",
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
    )

    report = runner.run(cases=[make_case()])

    result = report.results[0]

    assert result.status == "error"
    assert result.error


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("model", 123),
        ("analysis_id", 123),
    ],
)
def test_runner_rejects_invalid_analysis_identity_fields(
    field,
    invalid_value,
):
    """Modelo e identificador precisam respeitar o contrato de tipos."""

    analysis = make_analysis(
        [{"total_orders": 60}],
    )

    analysis[field] = invalid_value

    analyst = FakeAnalyst([analysis])

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
    )

    report = runner.run(cases=[make_case()])

    result = report.results[0]

    assert result.status == "error"
    assert result.error


def test_runner_resolves_missing_analysis_identity_fields():
    """A identidade pode ser completada com os dados do runner e do agente."""

    analysis = make_analysis(
        [{"total_orders": 60}],
    )

    analysis.pop("provider")
    analysis.pop("model")
    analysis.pop("analysis_id")

    analyst = FakeAnalyst([analysis])

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
        provider_name="gemini",
    )

    report = runner.run(cases=[make_case()])

    result = report.results[0]

    assert result.status == "passed"

    assert result.provider == "gemini"
    assert result.model == "fake-model"

    assert isinstance(result.analysis_id, str)
    assert result.analysis_id.strip()


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

    assert "token_cost_summary" in serialized


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


# ==========================================================
# CENÁRIOS COMPORTAMENTAIS
# ==========================================================


class BehavioralReferenceExecutor:
    """Executor de referência que registra qualquer tentativa de execução."""

    def __init__(self):
        self.executed_queries = []

    def execute(self, sql):
        self.executed_queries.append(sql)
        return []


class BehavioralAnalyst:
    """Agente falso para testar o fluxo comportamental sem chamadas externas."""

    model = "test-model"

    def __init__(self, analysis):
        self.analysis = analysis
        self.received_questions = []

    def analyze(self, question):
        self.received_questions.append(question)
        return self.analysis


class FailingBehaviorEvaluator:
    """Simula uma falha durante a avaliação comportamental."""

    def evaluate_behavior(self, case, analysis):
        raise RuntimeError("Falha simulada na avaliação comportamental.")


def make_behavioral_case(
    evaluation_type,
    expected_behavior,
    evaluation_criteria,
):
    """Converte um cenário SQL existente em um cenário comportamental."""

    return replace(
        make_case(),
        case_id=f"behavior_{evaluation_type}",
        reference_sql=None,
        required_columns=(),
        expected_analysis_mode=None,
        evaluation_type=evaluation_type,
        expected_behavior=expected_behavior,
        evaluation_criteria=evaluation_criteria,
    )


def make_behavioral_analysis(behavior_type):
    """Cria uma resposta estruturada do agente para cada comportamento."""

    behavior = {
        "type": behavior_type,
        "reason": ("Comportamento identificado a partir das informações disponíveis."),
        "clarification_question": None,
        "limitations": [],
        "supported_information": [],
        "sql_executed": False,
    }

    answer = "Análise comportamental de teste."

    if behavior_type == "ask_clarification":
        behavior["clarification_question"] = (
            "Você poderia especificar o período que deseja analisar?"
        )
        answer = "Preciso de uma informação adicional para responder corretamente."

    elif behavior_type == "abstain":
        behavior["limitations"] = [
            "Os dados disponíveis não permitem responder à pergunta."
        ]
        answer = "Não é possível responder com os dados disponíveis."

    elif behavior_type == "partial_answer":
        behavior["supported_information"] = [
            ("Existem informações suficientes para identificar uma parte do resultado.")
        ]
        behavior["limitations"] = [
            "Não existem dados suficientes para concluir toda a análise."
        ]
        answer = "Foi possível identificar parcialmente o resultado."

    return {
        "analysis_id": "agent-analysis-001",
        "provider": "gemini",
        "model": "test-model",
        "analysis_mode": "single_query",
        "answer": answer,
        "behavior": behavior,
        "results": [],
        "queries": [],
        "total_results": 0,
        "cache_hit": False,
        "steps": [
            {
                "name": "behavior_evaluation",
                "status": "success",
            },
        ],
        "partial_success": False,
    }


@pytest.mark.parametrize(
    (
        "evaluation_type",
        "expected_behavior",
        "evaluation_criteria",
    ),
    [
        (
            "clarification",
            "ask_clarification",
            (
                "clarification_question_present",
                "sql_not_executed",
            ),
        ),
        (
            "unanswerable",
            "abstain",
            (
                "explanation_present",
                "limitations_present",
                "sql_not_executed",
            ),
        ),
        (
            "partial_answer",
            "partial_answer",
            (
                "supported_information_present",
                "limitations_present",
                "sql_not_executed",
            ),
        ),
    ],
)
def test_behavioral_scenarios_skip_reference_sql(
    evaluation_type,
    expected_behavior,
    evaluation_criteria,
):
    """Cenários comportamentais não podem executar SQL de referência."""

    case = make_behavioral_case(
        evaluation_type=evaluation_type,
        expected_behavior=expected_behavior,
        evaluation_criteria=evaluation_criteria,
    )

    analysis = make_behavioral_analysis(expected_behavior)

    analyst = BehavioralAnalyst(analysis)
    reference_executor = BehavioralReferenceExecutor()

    runner = EvaluationRunner(
        provider_name="gemini",
        reference_executor=reference_executor,
        evaluator=ResultEvaluator(),
    )

    result = runner._run_case(
        case=case,
        analyst=analyst,
    )

    assert reference_executor.executed_queries == []

    assert analyst.received_questions == [case.question]

    assert result.status == "passed"
    assert result.score == 1.0

    assert result.metadata["evaluation_type"] == evaluation_type
    assert result.metadata["expected_behavior"] == expected_behavior
    assert result.metadata["observed_behavior"] == expected_behavior

    assert result.metadata["reference_sql_executed"] is False
    assert result.metadata["reference_sql_ms"] is None

    assert result.metadata["analysis_mode"] == "single_query"
    assert result.metadata["agent_analysis_id"] == "agent-analysis-001"

    assert result.metadata["llm_metrics"]["provider"] == "gemini"


def test_behavioral_evaluation_failure_returns_operational_error():
    """Falhas no avaliador comportamental devem ser isoladas pelo runner."""

    case = make_behavioral_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavioral_analysis("ask_clarification")

    analyst = BehavioralAnalyst(analysis)
    reference_executor = BehavioralReferenceExecutor()

    runner = EvaluationRunner(
        provider_name="gemini",
        reference_executor=reference_executor,
        evaluator=FailingBehaviorEvaluator(),
    )

    result = runner._run_case(
        case=case,
        analyst=analyst,
    )

    assert reference_executor.executed_queries == []

    assert result.status == "error"
    assert result.metadata["phase"] == "behavior_evaluation"

    assert result.metadata["reference_sql_executed"] is False
    assert result.metadata["reference_sql_ms"] is None

    assert result.error == ("Falha simulada na avaliação comportamental.")


def test_behavioral_report_counts_scenarios_correctly():
    """O relatório deve separar cenários SQL e comportamentais."""

    case = make_behavioral_case(
        evaluation_type="unanswerable",
        expected_behavior="abstain",
        evaluation_criteria=(
            "explanation_present",
            "limitations_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavioral_analysis("abstain")

    reference_executor = BehavioralReferenceExecutor()

    runner = EvaluationRunner(
        provider_name="gemini",
        analyst_factory=lambda provider_name: BehavioralAnalyst(analysis),
        reference_executor=reference_executor,
        evaluator=ResultEvaluator(),
    )

    report = runner.run(cases=[case])

    assert report.total_cases == 1

    assert report.metadata["scenario_count"] == 1
    assert report.metadata["sql_scenario_count"] == 0
    assert report.metadata["behavioral_scenario_count"] == 1

    assert len(report.results) == 1
    assert report.results[0].status == "passed"

    assert reference_executor.executed_queries == []


# ==========================================================
# AUXILIAR DE TESTE DE CUSTOS
# ==========================================================


def record_openai_cost_call(
    metrics: LLMMetrics,
    *,
    prompt_tokens: int = 1_000,
    output_tokens: int = 500,
    duration_ms: float = 100.0,
    model: str = "gpt-4.1-mini",
) -> None:
    """Registra uma chamada simulada com custo conhecido.

    Tarifas cadastradas:
    - Input: USD 0.40 por milhão de tokens.
    - Output: USD 1.60 por milhão de tokens.

    Para 1.000 tokens de entrada e 500 de saída:
    custo estimado = USD 0.0012.
    """

    metrics.record_success(
        model=model,
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        provider="openai",
    )


# ==========================================================
# PROPAGAÇÃO DE CUSTOS PARA RESULTADOS SQL
# ==========================================================


def test_runner_attaches_token_cost_to_successful_sql_result(monkeypatch):
    """O custo da análise deve chegar ao EvaluationCaseResult."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class CostTrackingAnalyst(FakeAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            record_openai_cost_call(metrics)

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "gpt-4.1-mini"

            return analysis

    analyst = CostTrackingAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
            ),
        ]
    )

    case = make_case()

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
        },
        provider_name="openai",
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "passed"
    assert result.token_cost_estimate is not None

    assert result.estimated_cost_usd == pytest.approx(0.0012)

    assert result.token_cost_estimate["currency"] == "USD"
    assert result.token_cost_estimate["requests_estimated"] == 1

    assert result.token_cost_estimate["requests_unestimated"] == 0

    assert result.token_cost_estimate["models"]["gpt-4.1-mini"][
        "estimated_cost_usd"
    ] == pytest.approx(0.0012)

    assert (
        result.metadata["llm_metrics"]["token_cost_estimate"]
        == result.token_cost_estimate
    )

    # Os snapshots precisam ser independentes.
    assert (
        result.metadata["llm_metrics"]["token_cost_estimate"]
        is not result.token_cost_estimate
    )


# ==========================================================
# PROPAGAÇÃO DE CUSTOS PARA CENÁRIOS COMPORTAMENTAIS
# ==========================================================


def test_runner_attaches_token_cost_to_behavioral_result(monkeypatch):
    """Cenários comportamentais também precisam preservar os custos."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class CostBehavioralAnalyst(BehavioralAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            record_openai_cost_call(metrics)

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "gpt-4.1-mini"

            return analysis

    case = make_behavioral_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavioral_analysis("ask_clarification")

    analyst = CostBehavioralAnalyst(analysis)

    reference_executor = BehavioralReferenceExecutor()

    runner = EvaluationRunner(
        provider_name="openai",
        reference_executor=reference_executor,
        evaluator=ResultEvaluator(),
    )

    result = runner._run_case(
        case=case,
        analyst=analyst,
    )

    assert result.status == "passed"

    assert reference_executor.executed_queries == []

    assert result.token_cost_estimate is not None

    assert result.estimated_cost_usd == pytest.approx(0.0012)

    assert result.token_cost_estimate["requests_estimated"] == 1


# ==========================================================
# PRESERVAÇÃO DE CUSTOS EM ERROS
# ==========================================================


def test_runner_preserves_cost_after_agent_execution_failure(
    monkeypatch,
):
    """Uma chamada bem-sucedida antes de uma falha não perde seu custo."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class CostThenFailAnalyst:
        model = "gpt-4.1-mini"

        def analyze(self, question: str) -> dict[str, Any]:
            record_openai_cost_call(metrics)

            raise RuntimeError("Falha após uma chamada de LLM.")

    runner = EvaluationRunner(
        provider_name="openai",
        analyst_factory=lambda **kwargs: CostThenFailAnalyst(),
        reference_executor=FakeReferenceExecutor(
            {
                "SELECT 1": [{"total_orders": 60}],
            }
        ),
    )

    report = runner.run(
        cases=[make_case()],
    )

    result = report.results[0]

    assert result.status == "error"

    assert result.metadata["phase"] == "agent_execution"

    assert result.token_cost_estimate is not None

    assert result.estimated_cost_usd == pytest.approx(0.0012)

    assert result.metadata["llm_metrics"]["requests"]["successful"] == 1


def test_runner_preserves_cost_when_result_extraction_fails(
    monkeypatch,
):
    """Erros de extração não devem apagar o custo da chamada."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class CostWithInvalidResultsAnalyst(FakeAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            record_openai_cost_call(metrics)

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "gpt-4.1-mini"
            analysis["results"] = "invalid-results"

            return analysis

    analyst = CostWithInvalidResultsAnalyst(
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
        provider_name="openai",
    )

    report = runner.run(cases=[make_case()])

    result = report.results[0]

    assert result.status == "error"

    assert result.metadata["phase"] == "result_extraction"

    assert result.token_cost_estimate is not None

    assert result.estimated_cost_usd == pytest.approx(0.0012)


# ==========================================================
# CUSTOS EM CENÁRIOS IGNORADOS
# ==========================================================


def test_runner_preserves_cost_when_multi_query_is_skipped(
    monkeypatch,
):
    """Um cenário ignorado depois da chamada ainda possui custo."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class CostMultiQueryAnalyst(FakeAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            record_openai_cost_call(metrics)

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "gpt-4.1-mini"

            return analysis

    analyst = CostMultiQueryAnalyst(
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

    case = make_case(
        required_columns=(
            "total_orders",
            "cancelled_orders",
        ),
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
        provider_name="openai",
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "skipped"

    assert result.error == "multi_query_results_not_combinable"

    assert result.token_cost_estimate is not None

    assert result.estimated_cost_usd == pytest.approx(0.0012)

    assert result.metadata["llm_metrics"]["requests"]["successful"] == 1


# ==========================================================
# AGREGAÇÃO DE CUSTOS POR MODELO E PROVEDOR
# ==========================================================


def test_report_aggregates_cost_by_model_and_provider(
    monkeypatch,
):
    """O relatório consolida custos sem misturar tarifas."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class MultipleCostAnalyst(FakeAnalyst):
        def __init__(
            self,
            responses: list[Any],
        ):
            super().__init__(responses)

            self.usage_batches = [
                (1_000, 500),
                (2_000, 1_000),
            ]

        def analyze(self, question: str) -> dict[str, Any]:
            prompt_tokens, output_tokens = self.usage_batches.pop(0)

            record_openai_cost_call(
                metrics,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "gpt-4.1-mini"

            return analysis

    cases = [
        make_case(
            case_id="cost-case-1",
            reference_sql="SELECT 1",
        ),
        make_case(
            case_id="cost-case-2",
            reference_sql="SELECT 2",
        ),
    ]

    analyst = MultipleCostAnalyst(
        [
            make_analysis(
                [{"total_orders": 60}],
            ),
            make_analysis(
                [{"total_orders": 80}],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [{"total_orders": 60}],
            "SELECT 2": [{"total_orders": 80}],
        },
        provider_name="openai",
    )

    report = runner.run(cases=cases)

    assert report.passed_cases == 2

    # Primeira chamada: USD 0.0012.
    # Segunda chamada: USD 0.0024.
    # Total: USD 0.0036.
    assert report.estimated_cost_usd == pytest.approx(0.0036)

    cost_summary = report.token_cost_summary

    assert cost_summary["currency"] == "USD"

    assert cost_summary["cases_with_cost_data"] == 2
    assert cost_summary["cases_without_cost_data"] == 0

    assert cost_summary["requests_estimated"] == 2
    assert cost_summary["requests_unestimated"] == 0

    assert cost_summary["is_complete"] is True

    assert cost_summary["estimated_cost_usd"] == pytest.approx(0.0036)

    provider_cost = cost_summary["providers"]["openai"]

    assert provider_cost["requests_total"] == 2
    assert provider_cost["requests_estimated"] == 2

    assert provider_cost["input_tokens"] == 3_000
    assert provider_cost["output_tokens"] == 1_500

    assert provider_cost["estimated_cost_usd"] == pytest.approx(0.0036)

    model_cost = cost_summary["models"]["openai/gpt-4.1-mini"]

    assert model_cost["provider"] == "openai"
    assert model_cost["model"] == "gpt-4.1-mini"

    assert model_cost["requests_total"] == 2
    assert model_cost["requests_estimated"] == 2

    assert model_cost["input_tokens"] == 3_000
    assert model_cost["output_tokens"] == 1_500

    assert model_cost["estimated_cost_usd"] == pytest.approx(0.0036)


def test_report_serialization_preserves_token_cost_details(
    monkeypatch,
):
    """A serialização preserva custos e detalhes dos modelos."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class CostAnalyst(FakeAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            record_openai_cost_call(metrics)

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "gpt-4.1-mini"

            return analysis

    analyst = CostAnalyst(
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
        provider_name="openai",
    )

    report = runner.run(
        cases=[make_case()],
        run_id="cost-serialization-test",
    )

    serialized = report.to_dict()

    assert serialized["run_id"] == "cost-serialization-test"

    assert serialized["token_cost_summary"]["currency"] == "USD"

    assert serialized["token_cost_summary"]["estimated_cost_usd"] == (
        pytest.approx(0.0012)
    )

    serialized_result = serialized["results"][0]

    assert serialized_result["token_cost_estimate"] is not None

    assert serialized_result["token_cost_estimate"][
        "estimated_cost_usd"
    ] == pytest.approx(0.0012)

    assert serialized_result["metadata"]["llm_metrics"]["token_cost_estimate"][
        "estimated_cost_usd"
    ] == pytest.approx(0.0012)


def test_report_marks_unknown_model_cost_as_incomplete(
    monkeypatch,
):
    """Modelos sem tarifa cadastrada não recebem custos inventados."""

    metrics = LLMMetrics()

    monkeypatch.setattr(
        "ai_analyst.evaluation.runner.llm_metrics",
        metrics,
    )

    class UnknownModelAnalyst(FakeAnalyst):
        def analyze(self, question: str) -> dict[str, Any]:
            metrics.record_success(
                model="model-not-in-pricing-table",
                duration_ms=100.0,
                prompt_tokens=1_000,
                output_tokens=500,
                provider="openai",
            )

            analysis = super().analyze(question)

            analysis["provider"] = "openai"
            analysis["model"] = "model-not-in-pricing-table"

            return analysis

    analyst = UnknownModelAnalyst(
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
        provider_name="openai",
    )

    report = runner.run(cases=[make_case()])

    result = report.results[0]

    assert result.status == "passed"

    assert result.token_cost_estimate is not None

    assert result.estimated_cost_usd is None

    cost_summary = report.token_cost_summary

    assert cost_summary["estimated_cost_usd"] is None

    assert cost_summary["requests_unestimated"] == 1

    assert cost_summary["is_complete"] is False


# ==========================================================
# VALIDAÇÃO DAS EVIDÊNCIAS EM CENÁRIOS SQL
# ==========================================================


def make_sql_evidence(
    *,
    query_id: str | int = 1,
    row_index: int = 0,
    column: str = "total_orders",
    value: Any = 100,
) -> dict[str, Any]:
    """Cria uma referência estruturada para um resultado SQL."""

    return {
        "query_id": query_id,
        "row_index": row_index,
        "column": column,
        "value": value,
    }


def make_sql_evidence_analysis(
    *,
    answer: str = "Foram encontrados 100 pedidos.",
    claim: str = "Foram encontrados 100 pedidos.",
    evidence_value: Any = 100,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cria uma análise SQL simples com fundamentação estruturada."""

    analysis = make_analysis(
        results=([{"total_orders": 100}] if results is None else results),
        answer=answer,
        answer_evidence=[
            {
                "claim": claim,
                "evidence": [
                    make_sql_evidence(
                        value=evidence_value,
                    )
                ],
            }
        ],
    )

    return analysis


def test_runner_validates_answer_evidence_for_sql_scenario():
    """Uma resposta SQL corretamente fundamentada deve ser aprovada."""

    case = make_case(
        required_columns=("total_orders",),
    )

    analyst = FakeAnalyst(
        [
            make_sql_evidence_analysis(),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 100},
            ],
        },
    )

    report = runner.run(
        cases=[case],
        run_id="sql-evidence-valid",
    )

    result = report.results[0]

    assert result.status == "passed"
    assert result.score == 1.0

    assert result.metadata["answer_evidence_status"] == "verified"

    assert result.metadata["answer_claims_supported"] is True

    evidence_report = result.metadata["answer_evidence_validation"]

    assert evidence_report["claims_count"] == 1
    assert evidence_report["valid_claims"] == 1
    assert evidence_report["valid_references"] == 1
    assert evidence_report["differences"] == []


def test_runner_rejects_sql_answer_with_invalid_evidence():
    """Uma resposta incorretamente fundamentada deve ser reprovada.

    A comparação SQL continua com seu score original.
    """

    case = make_case(
        required_columns=("total_orders",),
    )

    analyst = FakeAnalyst(
        [
            make_sql_evidence_analysis(
                answer="Foram encontrados 101 pedidos.",
                claim="Foram encontrados 101 pedidos.",
                evidence_value=101,
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 100},
            ],
        },
    )

    report = runner.run(
        cases=[case],
        run_id="sql-evidence-invalid",
    )

    result = report.results[0]

    assert report.total_cases == 1
    assert report.passed_cases == 0
    assert report.failed_cases == 1
    assert report.error_cases == 0

    assert result.status == "failed"

    # A comparação SQL estava correta.
    assert result.score == 1.0

    assert result.metadata["sql_result_status_before_evidence"] == "passed"

    assert result.metadata["sql_result_score_before_evidence"] == 1.0

    assert result.metadata["answer_evidence_status"] == "failed"

    assert result.metadata["answer_claims_supported"] is False

    assert any(
        "Fundamentação da resposta" in difference for difference in result.differences
    )


def test_runner_preserves_legacy_sql_analysis_without_evidence():
    """Análises antigas continuam funcionando sem evidências estruturadas."""

    case = make_case(
        required_columns=("total_orders",),
    )

    analyst = FakeAnalyst(
        [
            make_analysis(
                [
                    {"total_orders": 100},
                ],
            ),
        ]
    )

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 100},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "passed"
    assert result.score == 1.0

    assert result.metadata["answer_evidence_status"] == "not_evaluated"

    assert result.metadata["answer_claims_supported"] is False

    assert result.metadata["answer_evidence_validation"]["claims_count"] == 0


def test_runner_validates_evidence_from_multiple_sql_queries():
    """A fundamentação deve aceitar IDs de consultas compostas."""

    case = make_case(
        required_columns=("total_orders",),
        expected_analysis_mode="multi_query",
    )

    analysis = make_analysis(
        results=[],
        analysis_mode="multi_query",
        answer="Foram encontrados 100 pedidos.",
        queries=[
            {
                "id": "query-1",
                "status": "success",
                "sql_executed": True,
                "results": [
                    {"total_orders": 100},
                ],
            },
        ],
        answer_evidence=[
            {
                "claim": "Foram encontrados 100 pedidos.",
                "evidence": [
                    make_sql_evidence(
                        query_id="query-1",
                        column="total_orders",
                        value=100,
                    )
                ],
            }
        ],
    )

    analyst = FakeAnalyst([analysis])

    runner = make_runner(
        analyst=analyst,
        reference_responses={
            "SELECT 1": [
                {"total_orders": 100},
            ],
        },
    )

    report = runner.run(cases=[case])

    result = report.results[0]

    assert result.status == "passed"
    assert result.score == 1.0

    assert result.metadata["answer_evidence_status"] == "verified"

    assert result.metadata["answer_claims_supported"] is True

    assert result.metadata["answer_evidence_validation"]["valid_references"] == 1
