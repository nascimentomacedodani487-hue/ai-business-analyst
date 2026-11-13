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
