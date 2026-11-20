from __future__ import annotations

from math import inf, nan

import pytest

from ai_analyst.evaluation.evaluator import ResultEvaluator
from ai_analyst.evaluation.models import EvaluationCase


def make_case(**overrides):
    """Cria um cenário padrão para os testes."""

    data = {
        "case_id": "test_case",
        "question": "Qual é o valor total?",
        "reference_sql": "SELECT 10 AS total",
        "category": "test",
        "comparison_mode": "exact",
        "required_columns": ("total",),
    }

    data.update(overrides)

    return EvaluationCase(**data)


@pytest.fixture
def evaluator():
    return ResultEvaluator()


# ==========================================================
# COMPARAÇÃO BÁSICA
# ==========================================================


def test_identical_results_are_approved(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[{"total": 10}],
    )

    assert result.status == "passed"
    assert result.passed is True
    assert result.score == 1.0
    assert result.differences == []


def test_incorrect_value_is_rejected(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[{"total": 12}],
    )

    assert result.status == "failed"
    assert result.failed is True
    assert result.score < 1.0
    assert any("total" in difference for difference in result.differences)


def test_numeric_types_with_same_value_are_equal_exactly(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[{"total": 10.0}],
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_boolean_and_numeric_values_are_not_interchangeable(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": True}],
        actual_rows=[{"total": 1}],
    )

    assert result.status == "failed"
    assert result.score == 0.0


# ==========================================================
# TOLERÂNCIA NUMÉRICA
# ==========================================================


def test_numeric_tolerance_accepts_small_difference(evaluator):
    case = make_case(
        comparison_mode="tolerant",
        absolute_tolerance=0.01,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10.00}],
        actual_rows=[{"total": 10.005}],
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_numeric_tolerance_rejects_large_difference(evaluator):
    case = make_case(
        comparison_mode="tolerant",
        absolute_tolerance=0.01,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10.00}],
        actual_rows=[{"total": 10.02}],
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_relative_tolerance_is_applied(evaluator):
    case = make_case(
        comparison_mode="tolerant",
        absolute_tolerance=0,
        relative_tolerance=0.01,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 1000}],
        actual_rows=[{"total": 1009}],
    )

    assert result.status == "passed"


# ==========================================================
# VALORES NUMÉRICOS ESPECIAIS
# ==========================================================


@pytest.mark.parametrize(
    "comparison_mode",
    ["exact", "tolerant"],
)
def test_nan_values_are_equal_in_both_comparison_modes(
    evaluator,
    comparison_mode,
):
    case = make_case(
        comparison_mode=comparison_mode,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": nan}],
        actual_rows=[{"total": nan}],
    )

    assert result.status == "passed"
    assert result.score == 1.0
    assert result.differences == []


@pytest.mark.parametrize(
    "comparison_mode",
    ["exact", "tolerant"],
)
@pytest.mark.parametrize(
    ("expected_value", "actual_value"),
    [
        (inf, inf),
        (-inf, -inf),
    ],
)
def test_equal_infinite_values_are_accepted(
    evaluator,
    comparison_mode,
    expected_value,
    actual_value,
):
    case = make_case(
        comparison_mode=comparison_mode,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": expected_value}],
        actual_rows=[{"total": actual_value}],
    )

    assert result.status == "passed"
    assert result.score == 1.0


@pytest.mark.parametrize(
    "comparison_mode",
    ["exact", "tolerant"],
)
def test_nan_is_not_equal_to_finite_number(
    evaluator,
    comparison_mode,
):
    case = make_case(
        comparison_mode=comparison_mode,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": nan}],
        actual_rows=[{"total": 10}],
    )

    assert result.status == "failed"
    assert result.score == 0.0


# ==========================================================
# COMPARAÇÃO DE LINHAS
# ==========================================================


def test_unordered_rows_are_compared_correctly(evaluator):
    case = make_case(
        required_columns=("category", "count"),
        sort_rows=True,
    )

    expected = [
        {"category": "A", "count": 10},
        {"category": "B", "count": 20},
    ]

    actual = [
        {"category": "B", "count": 20},
        {"category": "A", "count": 10},
    ]

    result = evaluator.evaluate(
        case,
        expected_rows=expected,
        actual_rows=actual,
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_unordered_rows_preserve_duplicate_records(evaluator):
    case = make_case(
        required_columns=("category", "count"),
        sort_rows=True,
    )

    expected = [
        {"category": "A", "count": 10},
        {"category": "A", "count": 10},
        {"category": "B", "count": 20},
    ]

    actual = [
        {"category": "B", "count": 20},
        {"category": "A", "count": 10},
        {"category": "A", "count": 10},
    ]

    result = evaluator.evaluate(
        case,
        expected_rows=expected,
        actual_rows=actual,
    )

    assert result.status == "passed"
    assert result.score == 1.0
    assert result.differences == []


def test_unordered_rows_use_global_best_matching(evaluator):
    """A associação deve minimizar o total global de diferenças."""

    case = make_case(
        required_columns=("a", "b"),
        sort_rows=True,
    )

    expected = [
        {"a": 1, "b": 2},
        {"a": 0, "b": 0},
        {"a": 0, "b": 0},
    ]

    actual = [
        {"a": 0, "b": 2},
        {"a": 2, "b": 2},
        {"a": 0, "b": 1},
    ]

    result = evaluator.evaluate(
        case,
        expected_rows=expected,
        actual_rows=actual,
    )

    assert result.status == "failed"

    # A melhor associação global produz três diferenças
    # de células. Uma associação gulosa pode produzir quatro.
    assert len(result.differences) == 3


def test_ordered_rows_detect_different_positions(evaluator):
    case = make_case(
        required_columns=("category", "count"),
        sort_rows=False,
    )

    expected = [
        {"category": "A", "count": 10},
        {"category": "B", "count": 20},
    ]

    actual = [
        {"category": "B", "count": 20},
        {"category": "A", "count": 10},
    ]

    result = evaluator.evaluate(
        case,
        expected_rows=expected,
        actual_rows=actual,
    )

    assert result.status == "failed"
    assert result.score == 0.0


def test_missing_row_is_rejected(evaluator):
    case = make_case(
        required_columns=("category", "count"),
    )

    expected = [
        {"category": "A", "count": 10},
        {"category": "B", "count": 20},
    ]

    actual = [
        {"category": "A", "count": 10},
    ]

    result = evaluator.evaluate(
        case,
        expected_rows=expected,
        actual_rows=actual,
    )

    assert result.status == "failed"
    assert any("ausente" in difference for difference in result.differences)


def test_unexpected_row_is_rejected(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[
            {"total": 10},
            {"total": 20},
        ],
    )

    assert result.status == "failed"
    assert any("inesperada" in difference for difference in result.differences)


# ==========================================================
# ESQUEMA E COLUNAS
# ==========================================================


def test_missing_required_column_is_rejected(evaluator):
    case = make_case(
        required_columns=("total", "count"),
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10, "count": 5}],
        actual_rows=[{"total": 10}],
    )

    assert result.status == "failed"
    assert any("count" in difference for difference in result.differences)


def test_extra_column_is_rejected_by_default(evaluator):
    case = make_case(
        allow_extra_columns=False,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[{"total": 10, "unexpected": "extra"}],
    )

    assert result.status == "failed"
    assert any("unexpected" in difference for difference in result.differences)


def test_extra_column_can_be_allowed(evaluator):
    case = make_case(
        allow_extra_columns=True,
    )

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[{"total": 10, "additional": "information"}],
    )

    assert result.status == "passed"
    assert result.score == 1.0


# ==========================================================
# VALORES NULOS
# ==========================================================


def test_null_values_are_compared_correctly(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": None}],
        actual_rows=[{"total": None}],
    )

    assert result.status == "passed"

    different_result = evaluator.evaluate(
        case,
        expected_rows=[{"total": None}],
        actual_rows=[{"total": 10}],
    )

    assert different_result.status == "failed"


# ==========================================================
# SCORE
# ==========================================================


def test_score_measures_fully_matched_rows(evaluator):
    """O score representa linhas integralmente correspondentes."""

    case = make_case(
        required_columns=("a", "b", "c"),
        sort_rows=False,
    )

    expected = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
    ]

    actual = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 7, "b": 8, "c": 9},
    ]

    result = evaluator.evaluate(
        case,
        expected_rows=expected,
        actual_rows=actual,
    )

    assert result.status == "failed"
    assert result.score == 0.5


# ==========================================================
# RESULTADOS VAZIOS
# ==========================================================


def test_empty_results_are_approved_when_both_are_empty(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[],
        actual_rows=[],
    )

    assert result.status == "passed"
    assert result.score == 1.0
    assert result.metadata["schema_validation"] == "unavailable_empty_result"


# ==========================================================
# VALIDAÇÃO DOS ARGUMENTOS
# ==========================================================


def test_invalid_results_raise_type_error(evaluator):
    case = make_case()

    with pytest.raises(TypeError):
        evaluator.evaluate(
            case,
            expected_rows=[{"total": 10}],
            actual_rows="resultado inválido",
        )


def test_expected_rows_must_be_a_list(evaluator):
    case = make_case()

    with pytest.raises(
        TypeError,
        match="expected_rows deve ser uma lista",
    ):
        evaluator.evaluate(
            case,
            expected_rows=({"total": 10},),
            actual_rows=[{"total": 10}],
        )


def test_non_string_column_names_raise_type_error(evaluator):
    case = make_case()

    with pytest.raises(
        TypeError,
        match="nomes do tipo string",
    ):
        evaluator.evaluate(
            case,
            expected_rows=[{1: "total"}],
            actual_rows=[{1: "total"}],
        )


# ==========================================================
# METADADOS
# ==========================================================


def test_analysis_metadata_is_preserved(evaluator):
    case = make_case()

    result = evaluator.evaluate(
        case,
        expected_rows=[{"total": 10}],
        actual_rows=[{"total": 10}],
        analysis_id="analysis-123",
        provider="gemini",
        model="gemini-2.5-flash",
        duration_ms=1500.5,
        cache_hit=False,
    )

    assert result.analysis_id == "analysis-123"
    assert result.provider == "gemini"
    assert result.model == "gemini-2.5-flash"
    assert result.duration_ms == 1500.5
    assert result.cache_hit is False


# ==========================================================
# AVALIAÇÃO DE COMPORTAMENTOS
# ==========================================================


def make_behavior_case(
    evaluation_type: str,
    expected_behavior: str,
    evaluation_criteria: tuple[str, ...],
) -> EvaluationCase:
    """Cria um cenário comportamental sem SQL de referência."""

    from dataclasses import replace

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


def make_behavior_analysis(
    behavior_type: str,
    *,
    reason: str = "Justificativa comportamental de teste.",
    clarification_question: str | None = None,
    limitations: list[str] | None = None,
    supported_information: list[str] | None = None,
    sql_executed: bool = False,
    answer: str = "Resposta comportamental de teste.",
) -> dict:
    """Cria uma resposta do agente no contrato comportamental."""

    return {
        "answer": answer,
        "behavior": {
            "type": behavior_type,
            "reason": reason,
            "clarification_question": clarification_question,
            "limitations": limitations or [],
            "supported_information": supported_information or [],
            "sql_executed": sql_executed,
        },
    }


def test_evaluate_behavior_accepts_valid_clarification():
    case = make_behavior_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="ask_clarification",
        reason="A pergunta não informa o período da análise.",
        clarification_question="Qual período você deseja analisar?",
        answer="Preciso esclarecer o período para realizar a análise.",
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_evaluate_behavior_accepts_valid_abstention():
    case = make_behavior_case(
        evaluation_type="unanswerable",
        expected_behavior="abstain",
        evaluation_criteria=(
            "explanation_present",
            "limitations_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="abstain",
        reason="A base não possui informações de custo dos produtos.",
        limitations=["Não existe uma coluna de custo dos produtos no dataset."],
        answer=("Não é possível calcular o lucro com os dados disponíveis."),
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_evaluate_behavior_accepts_valid_partial_answer():
    case = make_behavior_case(
        evaluation_type="partial_answer",
        expected_behavior="partial_answer",
        evaluation_criteria=(
            "supported_information_present",
            "limitations_present",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="partial_answer",
        reason="A base permite responder parte da solicitação.",
        supported_information=[
            "É possível calcular a quantidade de pedidos entregues."
        ],
        limitations=[
            "Não existem informações suficientes para calcular a margem de lucro."
        ],
        answer=(
            "Foi possível identificar a quantidade de pedidos entregues, "
            "mas não calcular a margem de lucro."
        ),
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "passed"
    assert result.score == 1.0


def test_evaluate_behavior_rejects_wrong_expected_behavior():
    case = make_behavior_case(
        evaluation_type="unanswerable",
        expected_behavior="abstain",
        evaluation_criteria=(
            "explanation_present",
            "limitations_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="ask_clarification",
        clarification_question="Qual período deseja consultar?",
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_evaluate_behavior_rejects_clarification_without_question():
    case = make_behavior_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="ask_clarification",
        reason="A pergunta está ambígua.",
        clarification_question=None,
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_evaluate_behavior_rejects_abstention_without_limitations():
    case = make_behavior_case(
        evaluation_type="unanswerable",
        expected_behavior="abstain",
        evaluation_criteria=(
            "explanation_present",
            "limitations_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="abstain",
        reason="Não existem dados suficientes.",
        limitations=[],
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_evaluate_behavior_rejects_partial_answer_without_supported_information():
    case = make_behavior_case(
        evaluation_type="partial_answer",
        expected_behavior="partial_answer",
        evaluation_criteria=(
            "supported_information_present",
            "limitations_present",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="partial_answer",
        supported_information=[],
        limitations=["Parte das informações solicitadas não está disponível."],
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_evaluate_behavior_rejects_partial_answer_without_limitations():
    case = make_behavior_case(
        evaluation_type="partial_answer",
        expected_behavior="partial_answer",
        evaluation_criteria=(
            "supported_information_present",
            "limitations_present",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="partial_answer",
        supported_information=["A quantidade total de pedidos está disponível."],
        limitations=[],
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


@pytest.mark.parametrize(
    ("evaluation_type", "expected_behavior"),
    [
        ("clarification", "ask_clarification"),
        ("unanswerable", "abstain"),
    ],
)
def test_evaluate_behavior_rejects_sql_execution_when_not_allowed(
    evaluation_type: str,
    expected_behavior: str,
):
    criteria = (
        (
            "clarification_question_present",
            "sql_not_executed",
        )
        if expected_behavior == "ask_clarification"
        else (
            "explanation_present",
            "limitations_present",
            "sql_not_executed",
        )
    )

    case = make_behavior_case(
        evaluation_type=evaluation_type,
        expected_behavior=expected_behavior,
        evaluation_criteria=criteria,
    )

    analysis = make_behavior_analysis(
        behavior_type=expected_behavior,
        clarification_question=(
            "Qual período deseja consultar?"
            if expected_behavior == "ask_clarification"
            else None
        ),
        limitations=(
            ["Não existem dados suficientes."] if expected_behavior == "abstain" else []
        ),
        sql_executed=True,
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_evaluate_behavior_detects_results_even_when_sql_flag_is_false():
    case = make_behavior_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="ask_clarification",
        clarification_question="Qual período deseja consultar?",
        sql_executed=False,
    )

    # A evidência informa que não houve SQL, mas os resultados
    # presentes indicam que uma consulta foi executada.
    analysis["results"] = [
        {"total_orders": 60000},
    ]

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_evaluate_behavior_rejects_unknown_evaluation_criteria():
    case = make_behavior_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "criterio_inexistente",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="ask_clarification",
        clarification_question="Qual período deseja consultar?",
    )

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"
    assert result.score < 1.0


def test_sql_evaluator_rejects_behavioral_case():
    case = make_behavior_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    with pytest.raises(ValueError):
        ResultEvaluator().evaluate(
            case,
            expected_rows=[],
            actual_rows=[],
        )


# ==========================================================
# TESTES DE FUNDAMENTAÇÃO DAS RESPOSTAS
# ==========================================================


def make_answer_evidence(
    query_id=1,
    row_index=0,
    column="total",
    value=100,
):
    """Cria uma referência de evidência."""

    return {
        "query_id": query_id,
        "row_index": row_index,
        "column": column,
        "value": value,
    }


def make_answer_claim(
    claim="Foram encontrados 100 pedidos.",
    evidence=None,
):
    """Cria uma afirmação com evidências."""

    return {
        "claim": claim,
        "evidence": ([make_answer_evidence()] if evidence is None else evidence),
    }


def make_single_analysis_with_evidence(
    answer="Foram encontrados 100 pedidos.",
    evidence=None,
    claim="Foram encontrados 100 pedidos.",
):
    """Cria uma análise simples com evidências."""

    return {
        "question": "Quantos pedidos existem?",
        "sql": "SELECT COUNT(*) AS total FROM orders",
        "results": [
            {"total": 100},
        ],
        "answer": answer,
        "answer_evidence": [
            make_answer_claim(
                claim=claim,
                evidence=([make_answer_evidence()] if evidence is None else evidence),
            )
        ],
        "behavior": {
            "type": "answer",
            "sql_executed": True,
        },
    }


def test_answer_evidence_accepts_valid_single_query(evaluator):
    analysis = make_single_analysis_with_evidence()

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "verified"
    assert result["claims_supported"] is True
    assert result["claims_count"] == 1
    assert result["valid_claims"] == 1
    assert result["invalid_claims"] == 0
    assert result["reference_count"] == 1
    assert result["valid_references"] == 1
    assert result["invalid_references"] == 0
    assert result["differences"] == []


def test_answer_evidence_rejects_invented_value(evaluator):
    analysis = make_single_analysis_with_evidence(
        answer="Foram encontrados 101 pedidos.",
        evidence=[
            make_answer_evidence(value=101),
        ],
        claim="Foram encontrados 101 pedidos.",
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"
    assert result["claims_supported"] is False
    assert result["invalid_references"] == 1

    assert any("não corresponde" in difference for difference in result["differences"])


def test_answer_evidence_rejects_missing_query(evaluator):
    analysis = {
        "answer": "Foram encontrados 100 pedidos.",
        "queries": [
            {
                "id": 1,
                "status": "success",
                "results": [{"total": 100}],
            }
        ],
        "answer_evidence": [
            make_answer_claim(
                evidence=[
                    make_answer_evidence(query_id=99),
                ]
            )
        ],
    }

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert any(
        "consulta referenciada" in difference and "não existe" in difference
        for difference in result["differences"]
    )


def test_answer_evidence_rejects_failed_query(evaluator):
    analysis = {
        "answer": "Foram encontrados 100 pedidos.",
        "queries": [
            {
                "id": 1,
                "status": "failure",
                "sql_executed": True,
                "results": [{"total": 100}],
            }
        ],
        "answer_evidence": [
            make_answer_claim(),
        ],
    }

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert any(
        "não foi concluída" in difference for difference in result["differences"]
    )


def test_answer_evidence_rejects_missing_row(evaluator):
    analysis = make_single_analysis_with_evidence(
        evidence=[
            make_answer_evidence(row_index=5),
        ],
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert any(
        "linha referenciada" in difference for difference in result["differences"]
    )


def test_answer_evidence_rejects_missing_column(evaluator):
    analysis = make_single_analysis_with_evidence(
        evidence=[
            make_answer_evidence(column="unknown_column"),
        ],
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert any(
        "coluna referenciada" in difference for difference in result["differences"]
    )


def test_answer_evidence_rejects_claim_not_present_in_answer(evaluator):
    analysis = make_single_analysis_with_evidence()

    analysis["answer_evidence"][0]["claim"] = "Foram encontrados 200 pedidos."

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"
    assert result["claims_supported"] is False
    assert result["valid_claims"] == 0
    assert result["invalid_claims"] == 1

    assert any(
        "não foi localizada" in difference for difference in result["differences"]
    )


def test_answer_evidence_does_not_match_numeric_prefix(evaluator):
    """Uma afirmação com 100 não pode corresponder a uma resposta com 1000."""

    analysis = make_single_analysis_with_evidence(
        answer="Foram encontrados 1000 pedidos.",
        evidence=[
            make_answer_evidence(value=100),
        ],
        claim="Foram encontrados 100 pedidos.",
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"
    assert result["claims_supported"] is False
    assert result["valid_claims"] == 0
    assert result["invalid_claims"] == 1

    assert result["reference_count"] == 1
    assert result["valid_references"] == 1

    assert any(
        "não foi localizada" in difference for difference in result["differences"]
    )


@pytest.mark.parametrize(
    "queries",
    [
        None,
        [],
    ],
)
def test_answer_evidence_does_not_fallback_to_root_results_for_empty_multi_query(
    evaluator,
    queries,
):
    """Uma análise composta exige referências a consultas individuais."""

    analysis = make_single_analysis_with_evidence()

    analysis["analysis_mode"] = "multi_query"

    if queries is None:
        analysis.pop("queries", None)
    else:
        analysis["queries"] = queries

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"
    assert result["claims_supported"] is False
    assert result["valid_claims"] == 0
    assert result["invalid_claims"] == 1

    assert result["reference_count"] == 1
    assert result["valid_references"] == 0
    assert result["invalid_references"] == 1

    assert any("multi_query" in difference for difference in result["differences"])


def test_answer_evidence_rejects_numeric_answer_without_claim(evaluator):
    analysis = make_single_analysis_with_evidence(
        answer="Foram encontrados 100 pedidos e 20 restaurantes.",
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert "200" not in result["uncovered_numeric_values"]

    assert "20" in result["uncovered_numeric_values"]


def test_answer_evidence_rejects_numeric_claim_without_cell_support(evaluator):
    analysis = make_single_analysis_with_evidence(
        answer="Foram encontrados 100 pedidos.",
    )

    analysis["answer_evidence"][0]["claim"] = (
        "Foram encontrados 100 pedidos e 20 restaurantes."
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert any(
        "sem correspondência" in difference for difference in result["differences"]
    )


def test_answer_evidence_accepts_brazilian_thousands_format(evaluator):
    analysis = make_single_analysis_with_evidence(
        answer="Foram encontrados 1.200 pedidos.",
        evidence=[
            make_answer_evidence(
                value=1200,
            )
        ],
        claim="Foram encontrados 1.200 pedidos.",
    )

    analysis["results"] = [
        {"total": 1200},
    ]

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "verified"
    assert result["claims_supported"] is True
    assert result["valid_claims"] == 1
    assert result["valid_references"] == 1
    assert result["answer_numeric_values"] == ["1200"]
    assert result["unsupported_numeric_values"] == []
    assert result["uncovered_numeric_values"] == []
    assert result["differences"] == []


def test_answer_evidence_supports_multiple_queries(evaluator):
    analysis = {
        "question": "Analise pedidos e faturamento.",
        "answer": ("Foram encontrados 100 pedidos e faturamento de 5000."),
        "queries": [
            {
                "id": 1,
                "status": "success",
                "sql_executed": True,
                "results": [{"total_pedidos": 100}],
            },
            {
                "id": 2,
                "status": "success",
                "sql_executed": True,
                "results": [{"faturamento": 5000}],
            },
        ],
        "answer_evidence": [
            make_answer_claim(
                "Foram encontrados 100 pedidos.",
                [
                    make_answer_evidence(
                        query_id=1,
                        column="total_pedidos",
                        value=100,
                    )
                ],
            ),
            make_answer_claim(
                "Faturamento de 5000.",
                [
                    make_answer_evidence(
                        query_id=2,
                        column="faturamento",
                        value=5000,
                    )
                ],
            ),
        ],
    }

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "verified"
    assert result["claims_count"] == 2
    assert result["valid_claims"] == 2
    assert result["invalid_claims"] == 0
    assert result["valid_references"] == 2


def test_answer_evidence_does_not_accept_boolean_as_integer(evaluator):
    analysis = make_single_analysis_with_evidence(
        evidence=[
            make_answer_evidence(
                value=True,
            )
        ],
    )

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"


def test_answer_evidence_marks_legacy_analysis_not_evaluated(evaluator):
    analysis = {
        "answer": "Análise concluída.",
        "results": [{"total": 100}],
    }

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "not_evaluated"
    assert result["claims_supported"] is False


def test_answer_evidence_rejects_numeric_answer_without_evidence(evaluator):
    analysis = {
        "answer": "Foram encontrados 100 pedidos.",
        "results": [{"total": 100}],
        "answer_evidence": [],
    }

    result = evaluator.validate_answer_evidence(analysis)

    assert result["status"] == "failed"

    assert any(
        "não possui afirmações" in difference for difference in result["differences"]
    )


# ==========================================================
# TESTES DE CRITÉRIOS COMPORTAMENTAIS DE EVIDÊNCIAS
# ==========================================================


def make_evidence_behavior_case(
    criteria: tuple[str, ...],
):
    return make_behavior_case(
        evaluation_type="partial_answer",
        expected_behavior="partial_answer",
        evaluation_criteria=criteria,
    )


def make_evidence_behavior_analysis():
    analysis = make_behavior_analysis(
        behavior_type="partial_answer",
        supported_information=[
            "Foram encontrados 100 pedidos.",
        ],
        limitations=[
            "Não existem informações sobre custos.",
        ],
        answer="Foram encontrados 100 pedidos.",
    )

    analysis["results"] = [{"total": 100}]
    analysis["answer_evidence"] = [
        make_answer_claim(),
    ]

    return analysis


def test_evaluate_behavior_accepts_valid_answer_evidence():
    case = make_evidence_behavior_case(
        (
            "supported_information_present",
            "limitations_present",
            "answer_evidence_present",
            "answer_evidence_valid",
            "answer_claims_supported",
        )
    )

    analysis = make_evidence_behavior_analysis()

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "passed"
    assert result.score == 1.0

    report = result.metadata["answer_evidence_validation"]

    assert report["status"] == "verified"
    assert result.metadata["answer_evidence_status"] == "verified"
    assert result.metadata["answer_claims_supported"] is True


def test_evaluate_behavior_rejects_invalid_answer_evidence():
    case = make_evidence_behavior_case(
        (
            "supported_information_present",
            "limitations_present",
            "answer_evidence_present",
            "answer_evidence_valid",
            "answer_claims_supported",
        )
    )

    analysis = make_evidence_behavior_analysis()

    analysis["answer_evidence"][0]["evidence"][0]["value"] = 101

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"

    assert result.metadata["answer_evidence_validation"]["status"] == "failed"
    assert result.metadata["answer_evidence_status"] == "failed"
    assert result.metadata["answer_claims_supported"] is False


def test_evaluate_behavior_checks_sql_execution_in_failed_queries():
    case = make_behavior_case(
        evaluation_type="clarification",
        expected_behavior="ask_clarification",
        evaluation_criteria=(
            "clarification_question_present",
            "sql_not_executed",
        ),
    )

    analysis = make_behavior_analysis(
        behavior_type="ask_clarification",
        clarification_question="Qual período deseja consultar?",
        sql_executed=False,
    )

    analysis["queries"] = [
        {
            "id": 1,
            "status": "failure",
            "sql_executed": True,
            "results": [],
        }
    ]

    result = ResultEvaluator().evaluate_behavior(
        case,
        analysis,
    )

    assert result.status == "failed"

    assert any("execução SQL" in difference for difference in result.differences)
