from __future__ import annotations

from typing import Any

import pytest

from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.evaluation.dataset import load_evaluation_dataset
from ai_analyst.evaluation.models import EvaluationCase
from ai_analyst.evaluation.runner import EvaluationRunner

# ==========================================================
# FIXTURES
# ==========================================================


@pytest.fixture(scope="module")
def reference_executor():
    """Utiliza o SQLExecutor real e os guardrails existentes."""

    return SQLExecutor()


@pytest.fixture(scope="module")
def evaluation_cases() -> list[EvaluationCase]:
    """Carrega os dez cenários do dataset real."""

    return load_evaluation_dataset()


@pytest.fixture(scope="module")
def cases_by_id(
    evaluation_cases: list[EvaluationCase],
) -> dict[str, EvaluationCase]:
    return {case.case_id: case for case in evaluation_cases}


@pytest.fixture(scope="module")
def reference_results(
    evaluation_cases: list[EvaluationCase],
    reference_executor: SQLExecutor,
) -> dict[str, list[dict[str, Any]]]:
    """Executa todos os SQLs de referência no DuckDB real."""

    return {
        case.case_id: reference_executor.execute(case.reference_sql)
        for case in evaluation_cases
    }


# ==========================================================
# AGENTE SIMULADO PARA INTEGRAÇÃO
# ==========================================================


class ReplayAnalyst:
    """Reproduz resultados conhecidos sem chamar provedores de IA.

    Para cenários single_query, reproduz diretamente o resultado
    completo de referência.

    Para cenários multi_query, divide os resultados escalares
    em consultas independentes, permitindo testar a consolidação
    feita pelo EvaluationRunner.
    """

    model = "integration-replay-model"

    def __init__(
        self,
        cases: list[EvaluationCase],
        reference_results: dict[str, list[dict[str, Any]]],
    ):
        self.cases_by_question = {case.question: case for case in cases}

        self.reference_results = reference_results

        self.questions: list[str] = []

    def analyze(self, question: str) -> dict[str, Any]:
        self.questions.append(question)

        case = self.cases_by_question.get(question)

        if case is None:
            raise ValueError(f"Pergunta não encontrada no replay: {question}")

        rows = self.reference_results[case.case_id]

        # ------------------------------------------------------
        # ANÁLISE COMPOSTA
        # ------------------------------------------------------

        if case.expected_analysis_mode == "multi_query":
            if len(rows) != 1:
                raise ValueError(
                    "O replay multi_query espera exatamente "
                    f"uma linha de referência. "
                    f"Cenário: {case.case_id}."
                )

            reference_row = rows[0]

            missing_columns = set(case.required_columns) - set(reference_row)

            if missing_columns:
                raise ValueError(
                    "Colunas obrigatórias ausentes no resultado "
                    f"de referência de '{case.case_id}': "
                    f"{sorted(missing_columns)}."
                )

            queries = []

            for index, column in enumerate(
                case.required_columns,
                start=1,
            ):
                queries.append(
                    {
                        "id": f"replay-query-{index}",
                        "status": "success",
                        "results": [
                            {
                                column: reference_row[column],
                            },
                        ],
                    }
                )

            return {
                "analysis_id": f"replay-{len(self.questions)}",
                "provider": "gemini",
                "model": self.model,
                "analysis_mode": "multi_query",
                "queries": queries,
                "results": [],
                "total_results": sum(len(query["results"]) for query in queries),
                "cache_hit": False,
                "steps": [],
                "partial_success": False,
            }

        # ------------------------------------------------------
        # ANÁLISE SIMPLES
        # ------------------------------------------------------

        return {
            "analysis_id": f"replay-{len(self.questions)}",
            "provider": "gemini",
            "model": self.model,
            "analysis_mode": "single_query",
            "results": [dict(row) for row in rows],
            "total_results": len(rows),
            "cache_hit": False,
            "steps": [],
        }


# ==========================================================
# ESTRUTURA DO DATASET
# ==========================================================


def test_dataset_contains_ten_evaluation_cases(
    evaluation_cases: list[EvaluationCase],
):

    assert len(evaluation_cases) == 10

    case_ids = {case.case_id for case in evaluation_cases}

    assert case_ids == {
        "total_orders",
        "orders_by_status",
        "cancellation_rate",
        "financial_breakdown",
        "monthly_orders",
        "delivery_by_city",
        "ratings_by_category",
        "payment_distribution",
        "rain_delivery_comparison",
        "operational_overview",
    }


def test_operational_overview_requires_multi_query(
    cases_by_id: dict[str, EvaluationCase],
):

    case = cases_by_id["operational_overview"]

    assert case.expected_analysis_mode == "multi_query"

    assert set(case.required_columns) == {
        "total_orders",
        "cancellation_rate_pct",
        "total_discounts",
        "average_delivery_minutes",
    }


@pytest.mark.parametrize(
    "case_id",
    [
        "total_orders",
        "orders_by_status",
        "cancellation_rate",
        "financial_breakdown",
        "monthly_orders",
        "delivery_by_city",
        "ratings_by_category",
        "payment_distribution",
        "rain_delivery_comparison",
        "operational_overview",
    ],
)
def test_reference_sql_executes_through_real_guardrails(
    case_id: str,
    cases_by_id: dict[str, EvaluationCase],
    reference_results: dict[str, list[dict[str, Any]]],
):

    case = cases_by_id[case_id]
    rows = reference_results[case_id]

    assert isinstance(rows, list), f"{case_id}: resultado não é uma lista."

    assert rows, f"{case_id}: consulta não retornou registros."

    assert all(isinstance(row, dict) for row in rows)

    actual_columns = set().union(*(set(row.keys()) for row in rows))

    assert set(case.required_columns).issubset(actual_columns), (
        f"{case_id}: colunas obrigatórias ausentes. "
        f"Esperadas: {case.required_columns}. "
        f"Encontradas: {actual_columns}."
    )


# ==========================================================
# VOLUME E STATUS DOS PEDIDOS
# ==========================================================


def test_total_orders_matches_generated_dataset(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["total_orders"]

    assert len(rows) == 1
    assert rows[0]["total_orders"] == 60_000


def test_orders_by_status_matches_generated_dataset(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["orders_by_status"]

    counts = {row["status"]: row["order_count"] for row in rows}

    assert counts == {
        "canceled": 2_172,
        "delivered": 57_828,
    }

    assert sum(counts.values()) == 60_000


def test_cancellation_rate_is_consistent(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["cancellation_rate"]

    assert len(rows) == 1

    result = rows[0]

    assert result["canceled_orders"] == 2_172
    assert result["total_orders"] == 60_000

    assert result["cancellation_rate_pct"] == pytest.approx(
        3.62,
        abs=0.01,
    )


# ==========================================================
# INDICADORES FINANCEIROS
# ==========================================================


def test_financial_breakdown_is_consistent(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["financial_breakdown"]

    assert len(rows) == 1

    result = rows[0]

    gross = result["gross_items_value"]

    after_discount = result["items_value_after_discount"]

    discounts = result["total_discounts"]

    assert gross >= after_discount
    assert discounts >= 0

    assert gross == pytest.approx(
        after_discount + discounts,
        abs=0.02,
    )


# ==========================================================
# INDICADORES TEMPORAIS
# ==========================================================


def test_monthly_orders_returns_all_months_in_chronological_order(
    cases_by_id: dict[str, EvaluationCase],
    reference_results: dict[str, list[dict[str, Any]]],
):

    case = cases_by_id["monthly_orders"]

    rows = reference_results["monthly_orders"]

    months = [row["order_month"] for row in rows]

    expected_months = [f"2025-{month:02d}" for month in range(1, 13)]

    assert case.sort_rows is False

    assert months == expected_months

    assert sum(row["order_count"] for row in rows) == 60_000


# ==========================================================
# INDICADORES OPERACIONAIS
# ==========================================================


def test_delivery_by_city_returns_five_cities(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["delivery_by_city"]

    assert len(rows) == 5

    assert all(
        row["delivered_orders"] > 0 and row["average_delivery_minutes"] > 0
        for row in rows
    )


def test_ratings_by_category_returns_valid_averages(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["ratings_by_category"]

    assert len(rows) == 7

    for row in rows:
        assert row["rating_count"] > 0

        assert 1 <= row["average_rating"] <= 5


def test_payment_distribution_covers_all_orders(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["payment_distribution"]

    assert len(rows) == 5

    assert all(row["order_count"] > 0 for row in rows)

    assert sum(row["order_count"] for row in rows) == 60_000


def test_rain_comparison_contains_both_conditions(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["rain_delivery_comparison"]

    assert len(rows) == 2

    conditions = {row["is_raining"] for row in rows}

    assert conditions == {True, False}

    for row in rows:
        assert row["delivered_orders"] > 0

        assert row["average_delivery_minutes"] > 0


# ==========================================================
# PANORAMA GERAL DA OPERAÇÃO
# ==========================================================


def test_operational_overview_returns_consistent_indicators(
    reference_results: dict[str, list[dict[str, Any]]],
):

    rows = reference_results["operational_overview"]

    assert len(rows) == 1

    result = rows[0]

    assert result["total_orders"] == 60_000

    assert result["cancellation_rate_pct"] == pytest.approx(
        3.62,
        abs=0.01,
    )

    assert result["total_discounts"] >= 0

    assert result["average_delivery_minutes"] > 0


def test_operational_overview_matches_individual_reference_cases(
    reference_results: dict[str, list[dict[str, Any]]],
):

    overview = reference_results["operational_overview"][0]

    total_orders = reference_results["total_orders"][0]

    cancellation = reference_results["cancellation_rate"][0]

    financial = reference_results["financial_breakdown"][0]

    delivery = reference_results["delivery_by_city"]

    expected_average_delivery = sum(
        row["average_delivery_minutes"] * row["delivered_orders"] for row in delivery
    ) / sum(row["delivered_orders"] for row in delivery)

    assert overview["total_orders"] == (total_orders["total_orders"])

    assert overview["cancellation_rate_pct"] == (cancellation["cancellation_rate_pct"])

    assert overview["total_discounts"] == (financial["total_discounts"])

    assert overview["average_delivery_minutes"] == pytest.approx(
        expected_average_delivery,
        abs=0.02,
    )


# ==========================================================
# INTEGRAÇÃO DO RUNNER COM DUCKDB REAL
# ==========================================================


def test_runner_integrates_real_database_and_fake_analyst(
    evaluation_cases: list[EvaluationCase],
    reference_executor: SQLExecutor,
    reference_results: dict[str, list[dict[str, Any]]],
):

    created_analysts: list[ReplayAnalyst] = []

    def analyst_factory(**kwargs):
        analyst = ReplayAnalyst(
            cases=evaluation_cases,
            reference_results=reference_results,
        )

        created_analysts.append(analyst)

        return analyst

    runner = EvaluationRunner(
        provider_name="gemini",
        analyst_factory=analyst_factory,
        reference_executor=reference_executor,
    )

    report = runner.run(
        cases=evaluation_cases,
        run_id="duckdb-integration-test",
    )

    assert len(created_analysts) == 1

    analyst = created_analysts[0]

    assert len(analyst.questions) == 10

    assert report.run_id == "duckdb-integration-test"

    assert report.provider == "gemini"

    assert report.model == "integration-replay-model"

    assert report.total_cases == 10

    assert report.passed_cases == 10

    assert report.failed_cases == 0
    assert report.error_cases == 0
    assert report.skipped_cases == 0

    assert report.pass_rate == 100.0

    metrics = report.metadata["metrics"]

    assert metrics["summary"]["total_cases"] == 10

    assert metrics["summary"]["passed_cases"] == 10

    assert metrics["summary"]["pass_rate"] == 100.0

    assert len(metrics["by_category"]) == 10

    # ------------------------------------------------------
    # Verificar especificamente a execução multi_query.
    # ------------------------------------------------------

    overview_result = next(
        result for result in report.results if result.case_id == "operational_overview"
    )

    assert overview_result.status == "passed"

    assert overview_result.analysis_id == "replay-10"

    assert overview_result.expected_rows == (reference_results["operational_overview"])

    assert overview_result.actual_rows == (reference_results["operational_overview"])

    assert overview_result.metadata["analysis_mode"] == ("multi_query")

    assert overview_result.metadata["expected_analysis_mode"] == ("multi_query")

    assert overview_result.metadata["analysis_mode_matches"] is True

    assert overview_result.metadata["query_count"] == 4

    assert overview_result.metadata["successful_queries"] == 4

    assert overview_result.metadata["failed_queries"] == 0

    assert overview_result.metadata["results_combined"] is True

    assert overview_result.metadata["selected_query_ids"] == [
        "replay-query-1",
        "replay-query-2",
        "replay-query-3",
        "replay-query-4",
    ]
