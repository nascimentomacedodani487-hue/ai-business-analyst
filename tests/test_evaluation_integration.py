from __future__ import annotations

from typing import Any

import pytest

from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.evaluation.dataset import load_evaluation_dataset
from ai_analyst.evaluation.models import EvaluationCase
from ai_analyst.evaluation.runner import EvaluationRunner

# ==========================================================
# IDENTIFICADORES ESPERADOS DO DATASET
# ==========================================================

EXPECTED_SQL_CASE_IDS = (
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
    "top_restaurants_delivered",
    "combined_filters_q4",
    "category_vehicle_multi_join",
    "unrated_delivered_orders",
    "delivered_without_delivery_time",
    "top_categories_revenue",
)

EXPECTED_BEHAVIORAL_CASE_IDS = (
    "clarification_restaurant_performance",
    "unanswerable_net_profit",
    "partial_city_delivery_performance",
)

EXPECTED_CASE_IDS = EXPECTED_SQL_CASE_IDS + EXPECTED_BEHAVIORAL_CASE_IDS


# ==========================================================
# FIXTURES
# ==========================================================


@pytest.fixture(scope="module")
def reference_executor():
    """Utiliza o SQLExecutor real e os guardrails existentes."""

    return SQLExecutor()


@pytest.fixture(scope="module")
def evaluation_cases() -> list[EvaluationCase]:
    """Carrega os dezesseis cenários SQL e os três comportamentais."""

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
    """Executa somente os SQLs de referência dos cenários SQL."""

    results = {}

    for case in evaluation_cases:
        if case.evaluation_type != "sql_result":
            continue

        if not case.reference_sql:
            raise AssertionError(
                f"O cenário SQL '{case.case_id}' não possui SQL de referência."
            )

        results[case.case_id] = reference_executor.execute(case.reference_sql)

    return results


# ==========================================================
# AGENTE SIMULADO PARA INTEGRAÇÃO
# ==========================================================


class ReplayAnalyst:
    """Reproduz resultados conhecidos sem chamar provedores de IA.

    Para cenários SQL single_query, reproduz diretamente o resultado
    completo de referência.

    Para cenários SQL multi_query, divide os resultados escalares
    em consultas independentes, permitindo testar a consolidação
    feita pelo EvaluationRunner.

    Para cenários comportamentais, devolve evidências controladas
    correspondentes ao comportamento esperado pelo cenário.
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

    def _build_behavioral_analysis(
        self,
        case: EvaluationCase,
    ) -> dict[str, Any]:
        """Monta uma resposta simulada para um cenário comportamental."""

        behavior_type = case.expected_behavior

        behavior = {
            "type": behavior_type,
            "reason": "",
            "clarification_question": None,
            "limitations": [],
            "supported_information": [],
            "sql_executed": False,
        }

        if behavior_type == "ask_clarification":
            answer = (
                "Preciso de mais informações para determinar quais "
                "restaurantes tiveram o melhor desempenho."
            )

            behavior["reason"] = (
                "A expressão melhor desempenho não define uma métrica "
                "nem um período de comparação."
            )

            behavior["clarification_question"] = (
                "Você deseja comparar os restaurantes por quantidade "
                "de pedidos, faturamento bruto ou avaliação média? "
                "Qual período devemos considerar?"
            )

        elif behavior_type == "abstain":
            answer = (
                "Não é possível calcular o lucro líquido dos restaurantes "
                "com os dados disponíveis. O dataset não contém custos "
                "operacionais, salários, impostos ou despesas fixas."
            )

            behavior["reason"] = (
                "Os dados disponíveis não permitem calcular lucro líquido "
                "sem informações sobre custos e despesas."
            )

            behavior["limitations"] = [
                "Não existem dados de custos operacionais.",
                "Não existem dados de salários e impostos.",
                "Não existem dados de despesas fixas.",
            ]

        elif behavior_type == "partial_answer":
            answer = (
                "É possível comparar a quantidade de pedidos entregues "
                "e o tempo médio de entrega por cidade. Entretanto, "
                "não é possível determinar quanto dos atrasos foi causado "
                "pelo trânsito, pois essa informação não está registrada."
            )

            behavior["reason"] = (
                "Os indicadores de entrega estão disponíveis, mas o dataset "
                "não contém informações que permitam identificar o efeito "
                "do trânsito sobre os atrasos."
            )

            behavior["supported_information"] = [
                "Quantidade de pedidos entregues por cidade.",
                "Tempo médio de entrega por cidade.",
            ]

            behavior["limitations"] = [
                "O dataset não possui informações sobre condições de trânsito.",
                "Não é possível atribuir causalmente os atrasos ao trânsito.",
            ]

        else:
            raise ValueError(f"Comportamento não suportado no replay: {behavior_type}.")

        return {
            "analysis_id": f"replay-{len(self.questions)}",
            "provider": "gemini",
            "model": self.model,
            "analysis_mode": "single_query",
            "answer": answer,
            "behavior": behavior,
            "results": [],
            "queries": [],
            "total_results": 0,
            "cache_hit": False,
            "steps": [],
            "partial_success": False,
        }

    def analyze(self, question: str) -> dict[str, Any]:
        self.questions.append(question)

        case = self.cases_by_question.get(question)

        if case is None:
            raise ValueError(f"Pergunta não encontrada no replay: {question}")

        # ------------------------------------------------------
        # CENÁRIOS COMPORTAMENTAIS
        # ------------------------------------------------------

        if case.evaluation_type != "sql_result":
            return self._build_behavioral_analysis(case)

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


def test_dataset_contains_nineteen_evaluation_cases(
    evaluation_cases: list[EvaluationCase],
):
    assert len(evaluation_cases) == 19

    case_ids = {case.case_id for case in evaluation_cases}

    assert case_ids == set(EXPECTED_CASE_IDS)

    sql_cases = [
        case for case in evaluation_cases if case.evaluation_type == "sql_result"
    ]

    behavioral_cases = [
        case for case in evaluation_cases if case.evaluation_type != "sql_result"
    ]

    assert len(sql_cases) == 16
    assert len(behavioral_cases) == 3


def test_dataset_contains_expected_behavioral_scenarios(
    cases_by_id: dict[str, EvaluationCase],
):
    expected_behaviors = {
        "clarification_restaurant_performance": (
            "clarification",
            "ask_clarification",
        ),
        "unanswerable_net_profit": (
            "unanswerable",
            "abstain",
        ),
        "partial_city_delivery_performance": (
            "partial_answer",
            "partial_answer",
        ),
    }

    for case_id, (evaluation_type, expected_behavior) in expected_behaviors.items():
        case = cases_by_id[case_id]

        assert case.evaluation_type == evaluation_type
        assert case.expected_behavior == expected_behavior
        assert case.reference_sql is None


def test_dataset_contains_unique_case_ids(
    evaluation_cases: list[EvaluationCase],
):
    case_ids = [case.case_id for case in evaluation_cases]

    assert len(case_ids) == len(set(case_ids))


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
    EXPECTED_SQL_CASE_IDS,
)
def test_reference_sql_executes_through_real_guardrails(
    case_id: str,
    cases_by_id: dict[str, EvaluationCase],
    reference_results: dict[str, list[dict[str, Any]]],
):
    case = cases_by_id[case_id]
    rows = reference_results[case_id]

    assert case.evaluation_type == "sql_result"
    assert case.reference_sql is not None

    assert isinstance(rows, list), f"{case_id}: resultado não é uma lista."

    assert all(isinstance(row, dict) for row in rows)

    # ------------------------------------------------------
    # RESULTADO VAZIO É VÁLIDO NESTE CENÁRIO.
    # ------------------------------------------------------

    if case.category == "empty_result":
        assert "empty-result" in case.tags

        assert rows == []

        assert set(case.required_columns) == {
            "order_id",
            "city",
            "delivery_minutes",
        }

        return

    # ------------------------------------------------------
    # DEMAIS CENÁRIOS DEVEM RETORNAR REGISTROS.
    # ------------------------------------------------------

    assert rows, f"{case_id}: consulta não retornou registros."

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
# TOP N: RESTAURANTES COM MAIS PEDIDOS ENTREGUES
# ==========================================================


def test_top_restaurants_delivered_returns_five_ranked_restaurants(
    cases_by_id: dict[str, EvaluationCase],
    reference_results: dict[str, list[dict[str, Any]]],
):
    case = cases_by_id["top_restaurants_delivered"]

    rows = reference_results["top_restaurants_delivered"]

    assert len(rows) == 5

    assert case.sort_rows is False

    assert all(row["delivered_orders"] > 0 for row in rows)

    counts = [row["delivered_orders"] for row in rows]

    assert counts == sorted(counts, reverse=True)

    restaurant_ids = [row["restaurant_id"] for row in rows]

    assert len(restaurant_ids) == len(set(restaurant_ids))

    for previous, current in zip(rows, rows[1:]):
        if previous["delivered_orders"] == current["delivered_orders"]:
            assert previous["restaurant_id"] <= current["restaurant_id"]

    assert all(row["restaurant_name"] for row in rows)

    assert all(row["category"] for row in rows)

    assert all(row["city"] for row in rows)


# ==========================================================
# FILTROS COMBINADOS
# ==========================================================


def test_combined_filters_q4_returns_consistent_aggregates(
    reference_results: dict[str, list[dict[str, Any]]],
):
    rows = reference_results["combined_filters_q4"]

    assert len(rows) == 1

    result = rows[0]

    assert isinstance(result["delivered_orders"], int)

    assert result["delivered_orders"] >= 0

    if result["delivered_orders"] > 0:
        assert result["average_items_value"] > 0

        assert result["total_discounts"] >= 0

        assert result["average_delivery_minutes"] > 0
    else:
        assert result["average_items_value"] is None

        assert result["average_delivery_minutes"] is None


# ==========================================================
# MÚLTIPLOS JOINS E AGREGAÇÃO COMPOSTA
# ==========================================================


def test_category_vehicle_multi_join_returns_expected_groups(
    reference_results: dict[str, list[dict[str, Any]]],
):
    rows = reference_results["category_vehicle_multi_join"]

    assert len(rows) == 21

    assert all(row["category"] for row in rows)

    assert all(row["vehicle"] for row in rows)

    assert all(row["delivered_orders"] > 0 for row in rows)

    assert all(row["distinct_customers"] > 0 for row in rows)

    assert all(row["distinct_customers"] <= row["delivered_orders"] for row in rows)

    assert all(row["average_delivery_minutes"] > 0 for row in rows)

    groups = [(row["category"], row["vehicle"]) for row in rows]

    assert len(groups) == len(set(groups))

    assert groups == sorted(groups)


def test_category_vehicle_multi_join_covers_sao_paulo_deliveries(
    reference_results: dict[str, list[dict[str, Any]]],
):
    multi_join_rows = reference_results["category_vehicle_multi_join"]

    city_rows = reference_results["delivery_by_city"]

    sao_paulo = next(row for row in city_rows if row["city"] == "Sao Paulo")

    multi_join_total = sum(row["delivered_orders"] for row in multi_join_rows)

    assert multi_join_total == sao_paulo["delivered_orders"]


# ==========================================================
# TRATAMENTO DE VALORES NULL
# ==========================================================


def test_unrated_delivered_orders_handles_null_ratings(
    reference_results: dict[str, list[dict[str, Any]]],
):
    rows = reference_results["unrated_delivered_orders"]

    assert len(rows) == 1

    result = rows[0]

    assert result["delivered_orders"] == 57_828

    assert 0 <= result["unrated_delivered_orders"] <= 57_828

    expected_rate = (
        result["unrated_delivered_orders"] * 100.0 / result["delivered_orders"]
    )

    assert result["unrated_rate_pct"] == pytest.approx(
        expected_rate,
        abs=0.01,
    )


# ==========================================================
# RESULTADO VAZIO
# ==========================================================


def test_delivered_without_delivery_time_returns_empty_result(
    cases_by_id: dict[str, EvaluationCase],
    reference_results: dict[str, list[dict[str, Any]]],
):
    case = cases_by_id["delivered_without_delivery_time"]

    rows = reference_results["delivered_without_delivery_time"]

    assert case.category == "empty_result"

    assert "empty-result" in case.tags

    assert rows == []


# ==========================================================
# TOP N: CATEGORIAS POR VALOR BRUTO
# ==========================================================


def test_top_categories_revenue_returns_five_ranked_categories(
    cases_by_id: dict[str, EvaluationCase],
    reference_results: dict[str, list[dict[str, Any]]],
):
    case = cases_by_id["top_categories_revenue"]

    rows = reference_results["top_categories_revenue"]

    assert len(rows) == 5

    assert case.sort_rows is False

    assert all(row["category"] for row in rows)

    assert all(row["delivered_orders"] > 0 for row in rows)

    assert all(row["gross_items_value"] > 0 for row in rows)

    values = [row["gross_items_value"] for row in rows]

    assert values == sorted(values, reverse=True)

    for previous, current in zip(rows, rows[1:]):
        if previous["gross_items_value"] == current["gross_items_value"]:
            assert previous["category"] <= current["category"]

    categories = [row["category"] for row in rows]

    assert len(categories) == len(set(categories))


# ==========================================================
# INTEGRAÇÃO DO RUNNER COM DUCKDB REAL E CENÁRIOS MISTOS
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

    assert len(analyst.questions) == 19

    assert report.run_id == "duckdb-integration-test"

    assert report.provider == "gemini"

    assert report.model == "integration-replay-model"

    assert report.total_cases == 19

    assert report.passed_cases == 19

    assert report.failed_cases == 0
    assert report.error_cases == 0
    assert report.skipped_cases == 0

    assert report.pass_rate == 100.0

    assert report.metadata["scenario_count"] == 19
    assert report.metadata["sql_scenario_count"] == 16
    assert report.metadata["behavioral_scenario_count"] == 3

    metrics = report.metadata["metrics"]

    assert metrics["summary"]["total_cases"] == 19

    assert metrics["summary"]["passed_cases"] == 19

    assert metrics["summary"]["pass_rate"] == 100.0

    expected_category_count = len({case.category for case in evaluation_cases})

    assert len(metrics["by_category"]) == expected_category_count

    # ------------------------------------------------------
    # VERIFICAR ESPECIFICAMENTE A EXECUÇÃO MULTI_QUERY.
    # ------------------------------------------------------

    overview_result = next(
        result for result in report.results if result.case_id == "operational_overview"
    )

    assert overview_result.status == "passed"

    assert overview_result.analysis_id == "replay-10"

    assert overview_result.expected_rows == (reference_results["operational_overview"])

    assert overview_result.actual_rows == (reference_results["operational_overview"])

    assert overview_result.metadata["analysis_mode"] == "multi_query"

    assert overview_result.metadata["expected_analysis_mode"] == "multi_query"

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

    # ------------------------------------------------------
    # VERIFICAR O CENÁRIO DE RESULTADO VAZIO NO RUNNER.
    # ------------------------------------------------------

    empty_result = next(
        result
        for result in report.results
        if result.case_id == "delivered_without_delivery_time"
    )

    assert empty_result.status == "passed"

    assert empty_result.expected_rows == []

    assert empty_result.actual_rows == []

    assert empty_result.metadata["analysis_mode"] == "single_query"

    # ------------------------------------------------------
    # VERIFICAR OS CENÁRIOS COMPORTAMENTAIS NO RUNNER.
    # ------------------------------------------------------

    behavioral_results = {
        result.case_id: result
        for result in report.results
        if result.case_id in EXPECTED_BEHAVIORAL_CASE_IDS
    }

    assert set(behavioral_results) == set(EXPECTED_BEHAVIORAL_CASE_IDS)

    expected_behavior_by_case = {
        "clarification_restaurant_performance": "ask_clarification",
        "unanswerable_net_profit": "abstain",
        "partial_city_delivery_performance": "partial_answer",
    }

    for case_id, expected_behavior in expected_behavior_by_case.items():
        result = behavioral_results[case_id]

        assert result.status == "passed"

        assert result.metadata["evaluation_type"] != "sql_result"

        assert result.metadata["expected_behavior"] == expected_behavior

        assert result.metadata["observed_behavior"] == expected_behavior

        assert result.metadata["reference_sql_executed"] is False

        assert result.metadata["reference_sql_ms"] is None

        assert result.metadata["agent_analysis_id"] is not None

    # ------------------------------------------------------
    # VERIFICAR TODOS OS CENÁRIOS DA EXECUÇÃO.
    # ------------------------------------------------------

    report_case_ids = {result.case_id for result in report.results}

    assert report_case_ids == set(EXPECTED_CASE_IDS)
