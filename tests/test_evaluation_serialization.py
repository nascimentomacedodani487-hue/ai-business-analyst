from __future__ import annotations

import json
from typing import Any

import pytest

from ai_analyst.evaluation.cli import write_json_report
from ai_analyst.evaluation.dataset import load_evaluation_dataset
from ai_analyst.evaluation.models import EvaluationCase, EvaluationRunReport
from ai_analyst.evaluation.runner import EvaluationRunner

# ==========================================================
# CENÁRIOS COMPORTAMENTAIS
# ==========================================================

EXPECTED_BEHAVIORAL_CASES = {
    "clarification_restaurant_performance": "ask_clarification",
    "unanswerable_net_profit": "abstain",
    "partial_city_delivery_performance": "partial_answer",
}


# ==========================================================
# NORMALIZAÇÃO DOS VALORES ESPERADOS
# ==========================================================


def normalize_expected_text(value: Any) -> str:
    """Reproduz a normalização textual aplicada pelo ResultEvaluator."""

    return value.strip() if isinstance(value, str) else ""


# ==========================================================
# AGENTE SIMULADO
# ==========================================================


class SerializationReplayAnalyst:
    """Agente simulado para validar serialização sem chamadas de IA."""

    model = "serialization-test-model"

    def __init__(self, analysis: dict[str, Any]):
        self.analysis = analysis

    def analyze(self, question: str) -> dict[str, Any]:
        return self.analysis


class ForbiddenReferenceExecutor:
    """Impede a execução acidental de SQL de referência."""

    def execute(self, sql: str) -> list[dict[str, Any]]:
        raise AssertionError(
            "Cenários comportamentais não devem executar SQL de referência."
        )


# ==========================================================
# CONSTRUÇÃO DAS EVIDÊNCIAS
# ==========================================================


def make_behavioral_analysis(
    case: EvaluationCase,
) -> dict[str, Any]:
    """Cria uma resposta estruturada para cada comportamento."""

    behavior_type = case.expected_behavior

    behavior = {
        "type": behavior_type,
        "reason": "",
        "clarification_question": None,
        "limitations": [],
        "supported_information": [],
        "sql_executed": False,
    }

    results = []

    answer = ""

    # ------------------------------------------------------
    # ESCLARECIMENTO
    # ------------------------------------------------------

    if behavior_type == "ask_clarification":
        behavior["reason"] = (
            "A pergunta não especifica a métrica utilizada "
            "para determinar o melhor desempenho."
        )

        behavior["clarification_question"] = (
            "Você deseja comparar os restaurantes por quantidade "
            "de pedidos, faturamento bruto ou avaliação média?"
        )

        answer = (
            "Preciso esclarecer qual indicador deve ser utilizado "
            "para comparar o desempenho dos restaurantes."
        )

    # ------------------------------------------------------
    # ABSTENÇÃO
    # ------------------------------------------------------

    elif behavior_type == "abstain":
        behavior["reason"] = (
            "O dataset não contém as informações necessárias "
            "para calcular o lucro líquido."
        )

        behavior["limitations"] = [
            "Não existem dados de custos operacionais.",
            "Não existem dados de salários e impostos.",
            "Não existem dados de despesas fixas.",
        ]

        answer = "Não é possível calcular o lucro líquido com os dados disponíveis."

    # ------------------------------------------------------
    # RESPOSTA PARCIAL
    # ------------------------------------------------------

    elif behavior_type == "partial_answer":
        behavior["reason"] = (
            "Os indicadores de entrega estão disponíveis, "
            "mas não existem informações sobre o trânsito."
        )

        behavior["supported_information"] = [
            "Quantidade de pedidos entregues por cidade.",
            "Tempo médio de entrega por cidade.",
        ]

        behavior["limitations"] = [
            "O dataset não possui informações sobre condições de trânsito.",
            "Não é possível atribuir causalmente os atrasos ao trânsito.",
        ]

        behavior["sql_executed"] = True

        answer = (
            "É possível comparar a quantidade de pedidos entregues "
            "e o tempo médio de entrega por cidade. Entretanto, "
            "não é possível determinar quanto dos atrasos foi causado "
            "pelo trânsito."
        )

        results = [
            {
                "city": "Sao Paulo",
                "delivered_orders": 1000,
                "average_delivery_minutes": 35.5,
            }
        ]

    else:
        raise ValueError(f"Comportamento não suportado: {behavior_type}.")

    return {
        "analysis_id": f"serialization-{case.case_id}",
        "provider": "gemini",
        "model": "serialization-test-model",
        "analysis_mode": "single_query",
        "answer": answer,
        "behavior": behavior,
        "results": results,
        "queries": [],
        "total_results": len(results),
        "cache_hit": False,
        "steps": [
            {
                "name": "behavior_evaluation",
                "status": "success",
            }
        ],
        "partial_success": False,
    }


# ==========================================================
# FIXTURES
# ==========================================================


@pytest.fixture(scope="module")
def behavioral_cases() -> list[EvaluationCase]:
    """Carrega somente os três cenários comportamentais."""

    cases = load_evaluation_dataset()

    behavioral_cases = [case for case in cases if case.evaluation_type != "sql_result"]

    assert len(behavioral_cases) == 3

    assert {case.case_id for case in behavioral_cases} == set(EXPECTED_BEHAVIORAL_CASES)

    return behavioral_cases


# ==========================================================
# TESTES DE SERIALIZAÇÃO
# ==========================================================


@pytest.mark.parametrize(
    ("case_id", "expected_behavior"),
    EXPECTED_BEHAVIORAL_CASES.items(),
)
def test_behavioral_evidence_survives_json_serialization(
    case_id: str,
    expected_behavior: str,
    behavioral_cases: list[EvaluationCase],
    tmp_path,
):
    """Verifica a preservação das evidências no JSON final."""

    case = next(case for case in behavioral_cases if case.case_id == case_id)

    analysis = make_behavioral_analysis(case)

    analyst = SerializationReplayAnalyst(analysis)

    runner = EvaluationRunner(
        provider_name="gemini",
        analyst_factory=lambda **kwargs: analyst,
        reference_executor=ForbiddenReferenceExecutor(),
    )

    # ------------------------------------------------------
    # 1. EXECUTAR O CENÁRIO PELO RUNNER
    # ------------------------------------------------------

    report = runner.run(
        cases=[case],
        run_id=f"serialization-{case_id}",
    )

    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.failed_cases == 0
    assert report.error_cases == 0

    result = report.results[0]

    assert result.status == "passed"

    # ------------------------------------------------------
    # 2. SERIALIZAR O RELATÓRIO
    # ------------------------------------------------------

    payload = report.to_dict()

    assert payload["run_id"] == f"serialization-{case_id}"

    assert payload["summary"]["total_cases"] == 1

    assert len(payload["results"]) == 1

    serialized_result = payload["results"][0]

    assert serialized_result["case_id"] == case_id

    assert serialized_result["status"] == "passed"

    # ------------------------------------------------------
    # 3. VERIFICAR METADADOS COMPORTAMENTAIS
    # ------------------------------------------------------

    metadata = serialized_result["metadata"]

    assert metadata["evaluation_type"] == case.evaluation_type

    assert metadata["expected_behavior"] == expected_behavior

    assert metadata["observed_behavior"] == expected_behavior

    assert metadata["reference_sql_executed"] is False

    assert metadata["reference_sql_ms"] is None

    # ------------------------------------------------------
    # 4. VERIFICAR TODAS AS EVIDÊNCIAS
    # ------------------------------------------------------

    evidence = metadata["behavior_evidence"]

    original_evidence = analysis["behavior"]

    assert evidence["type"] == original_evidence["type"]

    assert evidence["reason"] == normalize_expected_text(original_evidence["reason"])

    assert evidence["clarification_question"] == normalize_expected_text(
        original_evidence["clarification_question"]
    )

    assert evidence["limitations"] == original_evidence["limitations"]

    assert (
        evidence["supported_information"] == original_evidence["supported_information"]
    )

    assert evidence["sql_executed"] == original_evidence["sql_executed"]

    assert metadata["answer"] == analysis["answer"]

    # ------------------------------------------------------
    # 5. VERIFICAR CRITÉRIOS E SCORE
    # ------------------------------------------------------

    checks = metadata["behavior_checks"]

    assert checks

    assert all(checks.values())

    assert metadata["passed_checks"] == metadata["total_checks"]

    assert metadata["behavior_pass_rate"] == 100.0

    assert serialized_result["score"] == 1.0

    # ------------------------------------------------------
    # 6. VERIFICAR RASTREABILIDADE
    # ------------------------------------------------------

    assert serialized_result["analysis_id"] == analysis["analysis_id"]

    assert serialized_result["provider"] == "gemini"

    assert serialized_result["model"] == "serialization-test-model"

    assert metadata["agent_analysis_id"] == analysis["analysis_id"]

    assert metadata["llm_metrics"]["provider"] == "gemini"

    # ------------------------------------------------------
    # 7. EXPORTAR JSON
    # ------------------------------------------------------

    output_path = tmp_path / f"{case_id}.json"

    saved_path = write_json_report(
        output_path,
        payload,
    )

    assert saved_path.exists()

    # ------------------------------------------------------
    # 8. REABRIR JSON
    # ------------------------------------------------------

    reloaded_payload = json.loads(saved_path.read_text(encoding="utf-8"))

    reloaded_result = reloaded_payload["results"][0]

    reloaded_metadata = reloaded_result["metadata"]

    reloaded_evidence = reloaded_metadata["behavior_evidence"]

    # ------------------------------------------------------
    # 9. COMPARAR DADOS ORIGINAIS E DADOS RECUPERADOS
    # ------------------------------------------------------

    assert reloaded_result["case_id"] == case_id

    assert reloaded_result["status"] == "passed"

    assert reloaded_result["score"] == 1.0

    assert reloaded_metadata["expected_behavior"] == expected_behavior

    assert reloaded_metadata["observed_behavior"] == expected_behavior

    assert reloaded_metadata["answer"] == analysis["answer"]

    assert reloaded_evidence == evidence

    assert reloaded_metadata["behavior_checks"] == checks

    assert reloaded_metadata["behavior_pass_rate"] == 100.0

    assert reloaded_metadata["reference_sql_executed"] is False


# ==========================================================
# TESTES DE INTEGRIDADE DO RELATÓRIO
# ==========================================================


def test_report_serialization_preserves_behavioral_dataset_metadata(
    behavioral_cases: list[EvaluationCase],
):
    """Verifica os metadados consolidados do relatório comportamental."""

    results = []

    for case in behavioral_cases:
        analysis = make_behavioral_analysis(case)

        analyst = SerializationReplayAnalyst(analysis)

        runner = EvaluationRunner(
            provider_name="gemini",
            analyst_factory=lambda **kwargs: analyst,
            reference_executor=ForbiddenReferenceExecutor(),
        )

        report = runner.run(
            cases=[case],
            run_id=f"metadata-{case.case_id}",
        )

        results.extend(report.results)

    consolidated_report = EvaluationRunReport(
        run_id="behavioral-dataset-serialization",
        provider="gemini",
        model="serialization-test-model",
        results=results,
        metadata={
            "scenario_count": 3,
            "sql_scenario_count": 0,
            "behavioral_scenario_count": 3,
        },
    )

    consolidated_report.finalize()

    payload = consolidated_report.to_dict()

    assert payload["metadata"]["scenario_count"] == 3

    assert payload["metadata"]["sql_scenario_count"] == 0

    assert payload["metadata"]["behavioral_scenario_count"] == 3

    assert payload["summary"]["total_cases"] == 3

    assert payload["summary"]["passed_cases"] == 3

    serialized_cases = {result["case_id"]: result for result in payload["results"]}

    assert set(serialized_cases) == set(EXPECTED_BEHAVIORAL_CASES)

    for case_id, expected_behavior in EXPECTED_BEHAVIORAL_CASES.items():
        result = serialized_cases[case_id]

        assert result["status"] == "passed"

        assert result["metadata"]["expected_behavior"] == expected_behavior

        assert result["metadata"]["observed_behavior"] == expected_behavior

        assert result["metadata"]["behavior_evidence"]["type"] == expected_behavior
