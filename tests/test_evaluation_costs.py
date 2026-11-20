"""Testes da integração de estimativa financeira do Evaluation Engine."""

import pytest

from ai_analyst.evaluation.metrics import calculate_evaluation_metrics
from ai_analyst.evaluation.models import (
    EvaluationCaseResult,
    EvaluationRunReport,
)


def make_snapshot(
    provider="gemini",
    model="gemini-2.5-flash",
    requests_total=1,
    input_tokens=1_000,
    output_tokens=500,
    thinking_tokens=0,
    tokens_unreported=0,
):
    """Cria um snapshot individual de consumo de LLM."""

    return {
        "analysis_id": "analysis-cost-test",
        "provider": provider,
        "model": model,
        "requests": {
            "total": requests_total,
            "successful": requests_total,
            "failed": 0,
            "retries": 0,
        },
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "thinking": thinking_tokens,
            "total": input_tokens + output_tokens + thinking_tokens,
            "unreported": tokens_unreported,
        },
        "latency": {
            "total_ms": 100.0,
            "average_ms": 100.0,
        },
    }


def make_report_cost_estimate(
    provider="gemini",
    model="gemini-2.5-flash",
    requests_total=1,
    requests_estimated=1,
    requests_unestimated=0,
    input_tokens=1_000,
    output_tokens=500,
    thinking_tokens=0,
    estimated_cost_usd=0.00155,
    is_complete=True,
    include_is_complete=True,
):
    """Cria uma estimativa financeira no formato do relatório consolidado."""

    model_data = {
        "provider": provider,
        "model": model,
        "requests_total": requests_total,
        "requests_estimated": requests_estimated,
        "requests_unestimated": requests_unestimated,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "pricing_tier": "standard",
        "pricing_source": "test_fixture",
        "pricing_as_of": "2026-11-16",
        "long_context_threshold_tokens": 128_000,
        "pricing_context_tiers": {
            "standard": requests_total,
        },
        "pricing_contexts": {
            "standard": {
                "requests": requests_total,
                "estimated_cost_usd": estimated_cost_usd,
                "rates": {
                    "input_per_million": 0.30,
                    "output_per_million": 2.50,
                },
            },
        },
    }

    estimate = {
        "provider": provider,
        "model": model,
        "currency": "USD",
        "pricing_as_of": "2026-11-16",
        "requests_estimated": requests_estimated,
        "requests_unestimated": requests_unestimated,
        "estimated_cost_usd": estimated_cost_usd,
        "models": {
            model: model_data,
        },
    }

    if include_is_complete:
        estimate["is_complete"] = is_complete

    return estimate


def make_result(
    case_id,
    provider="gemini",
    model="gemini-2.5-flash",
    category="volume",
    snapshot=None,
    status="passed",
    token_cost_estimate=None,
):
    """Cria um resultado de avaliação com métricas opcionais."""

    metadata = {"category": category}

    if snapshot is not None:
        metadata["llm_metrics"] = snapshot

    return EvaluationCaseResult(
        case_id=case_id,
        question=f"Question {case_id}",
        status=status,
        score=1.0 if status == "passed" else 0.0,
        provider=provider,
        model=model,
        metadata=metadata,
        token_cost_estimate=token_cost_estimate,
    )


# ==========================================================
# MÉTRICAS DE CUSTOS POR CENÁRIO, PROVEDOR E MODELO
# ==========================================================


def test_token_cost_estimate_calculates_cost_by_case_provider_and_model():
    """Calcula e agrega corretamente os custos de diferentes modelos."""

    results = [
        make_result(
            case_id="gemini-001",
            snapshot=make_snapshot(
                provider="gemini",
                model="gemini-2.5-flash",
                input_tokens=1_000,
                output_tokens=500,
                thinking_tokens=100,
            ),
        ),
        make_result(
            case_id="openai-001",
            provider="openai",
            model="gpt-4.1-mini",
            category="financial",
            snapshot=make_snapshot(
                provider="openai",
                model="gpt-4.1-mini",
                input_tokens=2_000,
                output_tokens=1_000,
            ),
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    usage = metrics["summary"]["llm_usage"]
    costs = usage["token_cost_estimate"]

    # ------------------------------------------------------
    # CONSOLIDAÇÃO GERAL
    # ------------------------------------------------------

    assert costs["basis"] == "standard_public_list_price"
    assert costs["currency"] == "USD"
    assert costs["not_actual_invoice"] is True

    assert costs["observed_cases"] == 2
    assert costs["estimated_cases"] == 2
    assert costs["incomplete_cases"] == 0

    # Gemini:
    # Entrada: 1.000 * 0,30 / 1.000.000 = 0,0003
    # Saída: (500 + 100 thinking) * 2,50 / 1.000.000 = 0,0015
    # Total = 0,0018
    #
    # OpenAI:
    # Entrada: 2.000 * 0,40 / 1.000.000 = 0,0008
    # Saída: 1.000 * 1,60 / 1.000.000 = 0,0016
    # Total = 0,0024

    assert costs["estimated_cost_usd"] == pytest.approx(0.0042)
    assert costs["known_subtotal_usd"] == pytest.approx(0.0042)

    # ------------------------------------------------------
    # CUSTOS POR CENÁRIO
    # ------------------------------------------------------

    assert len(costs["by_case"]) == 2

    gemini_case = costs["by_case"][0]
    openai_case = costs["by_case"][1]

    assert gemini_case["case_id"] == "gemini-001"
    assert gemini_case["status"] == "estimated"
    assert gemini_case["estimated_cost_usd"] == pytest.approx(0.0018)

    assert gemini_case["pricing"]["tokens"]["billable_output"] == 600

    assert openai_case["case_id"] == "openai-001"
    assert openai_case["status"] == "estimated"
    assert openai_case["estimated_cost_usd"] == pytest.approx(0.0024)

    # ------------------------------------------------------
    # CUSTOS POR PROVEDOR
    # ------------------------------------------------------

    assert costs["by_provider"]["gemini"]["estimated_cost_usd"] == pytest.approx(0.0018)

    assert costs["by_provider"]["openai"]["estimated_cost_usd"] == pytest.approx(0.0024)

    # ------------------------------------------------------
    # CUSTOS POR MODELO
    # ------------------------------------------------------

    gemini_model = costs["by_model"]["gemini/gemini-2.5-flash"]
    openai_model = costs["by_model"]["openai/gpt-4.1-mini"]

    assert gemini_model["estimated_cost_usd"] == pytest.approx(0.0018)
    assert openai_model["estimated_cost_usd"] == pytest.approx(0.0024)


def test_token_cost_estimate_preserves_category_grouping():
    """A estimativa também funciona nos relatórios segmentados."""

    results = [
        make_result(
            case_id="volume-001",
            category="volume",
            snapshot=make_snapshot(
                input_tokens=1_000,
                output_tokens=500,
            ),
        ),
        make_result(
            case_id="financial-001",
            category="financial",
            snapshot=make_snapshot(
                input_tokens=2_000,
                output_tokens=1_000,
            ),
        ),
    ]

    metrics = calculate_evaluation_metrics(results)

    volume_cost = metrics["by_category"]["volume"]["llm_usage"]["token_cost_estimate"]

    financial_cost = metrics["by_category"]["financial"]["llm_usage"][
        "token_cost_estimate"
    ]

    assert volume_cost["estimated_cost_usd"] == pytest.approx(0.00155)
    assert financial_cost["estimated_cost_usd"] == pytest.approx(0.0031)


def test_unknown_model_produces_incomplete_cost_estimate():
    """Modelos sem tarifas cadastradas não recebem custos inventados."""

    result = make_result(
        case_id="unknown-001",
        provider="openai",
        model="future-model-unknown",
        snapshot=make_snapshot(
            provider="openai",
            model="future-model-unknown",
        ),
    )

    usage = calculate_evaluation_metrics([result])["summary"]["llm_usage"]
    costs = usage["token_cost_estimate"]

    assert costs["estimated_cost_usd"] is None
    assert costs["known_subtotal_usd"] == 0.0

    assert costs["estimated_cases"] == 0
    assert costs["incomplete_cases"] == 1

    case = costs["by_case"][0]

    assert case["status"] == "incomplete"
    assert case["reason"] == "model_not_priced"
    assert case["estimated_cost_usd"] is None

    model = costs["by_model"]["openai/future-model-unknown"]

    assert model["estimated_cost_usd"] is None
    assert model["incomplete_cases"] == 1


def test_unreported_tokens_preserve_known_subtotal_without_claiming_total():
    """Tokens conhecidos geram subtotal, mas não um total incompleto."""

    result = make_result(
        case_id="partial-001",
        snapshot=make_snapshot(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=1_000,
            output_tokens=500,
            tokens_unreported=1,
        ),
    )

    usage = calculate_evaluation_metrics([result])["summary"]["llm_usage"]
    costs = usage["token_cost_estimate"]

    # Entrada: 0,0003
    # Saída: 0,00125
    # Subtotal conhecido: 0,00155

    assert costs["estimated_cost_usd"] is None
    assert costs["known_subtotal_usd"] == pytest.approx(0.00155)

    assert costs["estimated_cases"] == 0
    assert costs["incomplete_cases"] == 1

    case = costs["by_case"][0]

    assert case["status"] == "incomplete"
    assert case["reason"] == "some_requests_without_token_usage"
    assert case["estimated_cost_usd"] is None
    assert case["known_subtotal_usd"] == pytest.approx(0.00155)


def test_zero_usage_produces_zero_cost():
    """Cenários sem chamadas e sem tokens possuem custo conhecido zero."""

    result = make_result(
        case_id="zero-usage",
        snapshot=make_snapshot(
            requests_total=0,
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            tokens_unreported=0,
        ),
    )

    costs = calculate_evaluation_metrics([result])["summary"]["llm_usage"][
        "token_cost_estimate"
    ]

    assert costs["observed_cases"] == 1
    assert costs["estimated_cases"] == 1
    assert costs["incomplete_cases"] == 0

    assert costs["estimated_cost_usd"] == 0.0
    assert costs["known_subtotal_usd"] == 0.0


def test_results_without_llm_snapshots_do_not_generate_fake_costs():
    """Resultados legados sem snapshots não recebem estimativas artificiais."""

    result = make_result(
        case_id="legacy-001",
        snapshot=None,
    )

    usage = calculate_evaluation_metrics([result])["summary"]["llm_usage"]
    costs = usage["token_cost_estimate"]

    assert usage["observed_cases"] == 0

    assert costs["observed_cases"] == 0
    assert costs["estimated_cases"] == 0
    assert costs["incomplete_cases"] == 0

    assert costs["estimated_cost_usd"] is None
    assert costs["known_subtotal_usd"] == 0.0
    assert costs["by_case"] == []
    assert costs["by_provider"] == {}
    assert costs["by_model"] == {}


def test_duplicate_case_ids_are_preserved_in_cost_report():
    """Cenários repetidos não devem sobrescrever resultados financeiros."""

    results = [
        make_result(
            case_id="repeated-case",
            snapshot=make_snapshot(
                input_tokens=1_000,
                output_tokens=500,
            ),
        ),
        make_result(
            case_id="repeated-case",
            snapshot=make_snapshot(
                input_tokens=2_000,
                output_tokens=1_000,
            ),
        ),
    ]

    costs = calculate_evaluation_metrics(results)["summary"]["llm_usage"][
        "token_cost_estimate"
    ]

    assert len(costs["by_case"]) == 2

    assert costs["by_case"][0]["case_id"] == "repeated-case"
    assert costs["by_case"][1]["case_id"] == "repeated-case"

    assert costs["estimated_cases"] == 2
    assert costs["estimated_cost_usd"] == pytest.approx(0.00465)


def test_legacy_pricing_fields_remain_compatible():
    """Os campos históricos continuam preservando seus contratos."""

    result = make_result(
        case_id="legacy-pricing-001",
        snapshot=make_snapshot(
            provider="gemini",
            model="gemini-2.5-flash",
        ),
    )

    usage = calculate_evaluation_metrics([result])["summary"]["llm_usage"]

    assert usage["pricing_tier"] == "free"
    assert usage["cost_estimate_usd"] == 0.0

    assert usage["pricing_by_provider"]["gemini"]["cost_estimate_usd"] == 0.0

    # A nova estimativa representa o preço público padrão,
    # independentemente do campo histórico de nível gratuito.
    assert usage["token_cost_estimate"]["basis"] == "standard_public_list_price"

    assert usage["token_cost_estimate"]["estimated_cost_usd"] == pytest.approx(0.00155)


# ==========================================================
# AGREGAÇÃO FINANCEIRA DO RELATÓRIO CONSOLIDADO
# ==========================================================


def test_report_aggregates_costs_from_all_evaluation_statuses():
    """Todos os status contribuem para o consumo financeiro."""

    results = [
        make_result(
            case_id="passed-gemini",
            status="passed",
            token_cost_estimate=make_report_cost_estimate(
                provider="gemini",
                model="gemini-2.5-flash",
                estimated_cost_usd=0.001,
            ),
        ),
        make_result(
            case_id="failed-gemini",
            status="failed",
            token_cost_estimate=make_report_cost_estimate(
                provider="gemini",
                model="gemini-2.5-flash",
                estimated_cost_usd=0.002,
            ),
        ),
        make_result(
            case_id="error-openai",
            provider="openai",
            model="gpt-4.1-mini",
            status="error",
            token_cost_estimate=make_report_cost_estimate(
                provider="openai",
                model="gpt-4.1-mini",
                estimated_cost_usd=0.003,
            ),
        ),
        make_result(
            case_id="skipped-gemini",
            status="skipped",
            token_cost_estimate=make_report_cost_estimate(
                provider="gemini",
                model="gemini-2.5-flash",
                estimated_cost_usd=0.004,
            ),
        ),
    ]

    report = EvaluationRunReport(
        run_id="all-statuses-cost-test",
        results=results,
    )

    costs = report.token_cost_summary

    # ------------------------------------------------------
    # CONTAGEM DOS RESULTADOS
    # ------------------------------------------------------

    assert report.total_cases == 4
    assert report.passed_cases == 1
    assert report.failed_cases == 1
    assert report.error_cases == 1
    assert report.skipped_cases == 1

    # skipped continua fora do denominador da taxa de aprovação.
    assert report.evaluated_cases == 3
    assert report.pass_rate == pytest.approx(33.33)

    # ------------------------------------------------------
    # CONSOLIDAÇÃO FINANCEIRA
    # ------------------------------------------------------

    assert costs["cases_with_cost_data"] == 4
    assert costs["cases_without_cost_data"] == 0

    assert costs["requests_estimated"] == 4
    assert costs["requests_unestimated"] == 0

    assert costs["estimated_cost_usd"] == pytest.approx(0.010)
    assert costs["is_complete"] is True

    assert report.estimated_cost_usd == pytest.approx(0.010)

    # ------------------------------------------------------
    # CONSOLIDAÇÃO POR PROVEDOR
    # ------------------------------------------------------

    assert costs["providers"]["gemini"]["estimated_cost_usd"] == pytest.approx(0.007)

    assert costs["providers"]["gemini"]["requests_total"] == 3

    assert costs["providers"]["openai"]["estimated_cost_usd"] == pytest.approx(0.003)

    assert costs["providers"]["openai"]["requests_total"] == 1

    # ------------------------------------------------------
    # CONSOLIDAÇÃO POR MODELO
    # ------------------------------------------------------

    gemini_model = costs["models"]["gemini/gemini-2.5-flash"]
    openai_model = costs["models"]["openai/gpt-4.1-mini"]

    assert gemini_model["estimated_cost_usd"] == pytest.approx(0.007)
    assert gemini_model["requests_total"] == 3

    assert openai_model["estimated_cost_usd"] == pytest.approx(0.003)
    assert openai_model["requests_total"] == 1


def test_skipped_result_without_cost_data_makes_report_incomplete():
    """Um cenário skipped sem estimativa não pode desaparecer do relatório."""

    results = [
        make_result(
            case_id="known-cost",
            status="passed",
            token_cost_estimate=make_report_cost_estimate(
                estimated_cost_usd=0.002,
            ),
        ),
        make_result(
            case_id="skipped-without-cost",
            status="skipped",
            token_cost_estimate=None,
        ),
    ]

    report = EvaluationRunReport(
        run_id="skipped-without-cost-test",
        results=results,
    )

    costs = report.token_cost_summary

    assert costs["cases_with_cost_data"] == 1
    assert costs["cases_without_cost_data"] == 1

    # O subtotal conhecido é preservado.
    assert costs["estimated_cost_usd"] == pytest.approx(0.002)

    # Porém não é possível afirmar que o total da execução está completo.
    assert costs["is_complete"] is False


def test_report_without_explicit_snapshot_completeness_is_incomplete():
    """Snapshots sem is_complete não são considerados completos."""

    estimate = make_report_cost_estimate(
        estimated_cost_usd=0.002,
        include_is_complete=False,
    )

    result = make_result(
        case_id="missing-completeness",
        token_cost_estimate=estimate,
    )

    report = EvaluationRunReport(
        run_id="missing-completeness-test",
        results=[result],
    )

    costs = report.token_cost_summary

    assert costs["cases_with_cost_data"] == 1
    assert costs["cases_without_cost_data"] == 0

    # O custo continua contabilizado.
    assert costs["estimated_cost_usd"] == pytest.approx(0.002)

    # A ausência de confirmação impede declarar completude.
    assert costs["is_complete"] is False


def test_report_accepts_explicitly_complete_cost_snapshot():
    """Snapshots que declaram is_complete=True podem fechar a estimativa."""

    result = make_result(
        case_id="explicitly-complete",
        token_cost_estimate=make_report_cost_estimate(
            estimated_cost_usd=0.002,
            is_complete=True,
        ),
    )

    report = EvaluationRunReport(
        run_id="explicit-completeness-test",
        results=[result],
    )

    costs = report.token_cost_summary

    assert costs["cases_with_cost_data"] == 1
    assert costs["cases_without_cost_data"] == 0

    assert costs["requests_estimated"] == 1
    assert costs["requests_unestimated"] == 0

    assert costs["estimated_cost_usd"] == pytest.approx(0.002)
    assert costs["is_complete"] is True


def test_report_preserves_known_cost_when_another_request_is_unestimated():
    """Solicitações sem estimativa preservam o subtotal conhecido."""

    result = make_result(
        case_id="partially-estimated",
        token_cost_estimate=make_report_cost_estimate(
            requests_total=2,
            requests_estimated=1,
            requests_unestimated=1,
            estimated_cost_usd=0.002,
            is_complete=False,
        ),
    )

    report = EvaluationRunReport(
        run_id="partially-estimated-test",
        results=[result],
    )

    costs = report.token_cost_summary

    assert costs["requests_estimated"] == 1
    assert costs["requests_unestimated"] == 1

    assert costs["estimated_cost_usd"] == pytest.approx(0.002)
    assert costs["is_complete"] is False

    model = costs["models"]["gemini/gemini-2.5-flash"]

    assert model["requests_total"] == 2
    assert model["requests_estimated"] == 1
    assert model["requests_unestimated"] == 1


def test_report_preserves_explicit_zero_cost_as_valid_estimate():
    """Uma estimativa explicitamente igual a zero continua sendo válida."""

    result = make_result(
        case_id="zero-cost-report",
        status="skipped",
        token_cost_estimate=make_report_cost_estimate(
            requests_total=0,
            requests_estimated=0,
            requests_unestimated=0,
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            estimated_cost_usd=0.0,
            is_complete=True,
        ),
    )

    report = EvaluationRunReport(
        run_id="zero-cost-report-test",
        results=[result],
    )

    costs = report.token_cost_summary

    assert costs["cases_with_cost_data"] == 1
    assert costs["cases_without_cost_data"] == 0

    assert costs["estimated_cost_usd"] == 0.0
    assert report.estimated_cost_usd == 0.0

    assert costs["is_complete"] is True

    model = costs["models"]["gemini/gemini-2.5-flash"]

    assert model["estimated_cost_usd"] == 0.0
