"""Métricas consolidadas do AI Evaluation Engine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal
from math import isfinite
from statistics import fmean
from typing import Any

from ai_analyst.evaluation.models import EvaluationCaseResult
from ai_analyst.llm.metrics import PROVIDER_MODELS, get_pricing_metadata
from ai_analyst.llm.pricing import estimate_token_cost

# ==========================================================
# ADERÊNCIA AO PLANEJAMENTO
# ==========================================================


def _summarize_analysis_mode_adherence(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Calcula a aderência aos modos de análise esperados."""

    expected_cases = 0
    observed_cases = 0
    matches = 0
    mismatches = 0

    for result in results:
        metadata = result.metadata

        if not isinstance(metadata, Mapping):
            continue

        expected_mode = metadata.get("expected_analysis_mode")

        if expected_mode is None:
            continue

        expected_cases += 1
        actual_mode = metadata.get("analysis_mode")

        if not isinstance(actual_mode, str) or not actual_mode.strip():
            continue

        observed_cases += 1

        if actual_mode == expected_mode:
            matches += 1
        else:
            mismatches += 1

    unobserved_cases = expected_cases - observed_cases

    adherence_rate = (
        round(matches / observed_cases * 100, 2) if observed_cases else None
    )

    return {
        "expected_cases": expected_cases,
        "observed_cases": observed_cases,
        "unobserved_cases": unobserved_cases,
        "matches": matches,
        "mismatches": mismatches,
        "rate": adherence_rate,
    }


# ==========================================================
# VALIDAÇÃO DE MÉTRICAS
# ==========================================================


def _is_valid_metric_count(value: Any) -> bool:
    """Verifica se um contador é um inteiro não negativo válido."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_valid_metric_number(value: Any) -> bool:
    """Verifica se uma métrica numérica é finita e não negativa."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False

    try:
        return isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def _round_cost(value: Decimal) -> float:
    """Arredonda valores financeiros preservando precisão decimal."""

    return float(
        value.quantize(
            Decimal("0.000000000001"),
            rounding=ROUND_HALF_UP,
        )
    )


# ==========================================================
# QUALIDADE DOS RESULTADOS
# ==========================================================


def _summarize_result_quality(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Consolida as métricas de correspondência de linhas e células.

    cell_accuracy representa a proporção de células corretas
    dentro das linhas efetivamente comparadas.

    Linhas inesperadas continuam afetando a avaliação estrutural
    e o status do cenário, mas não são transformadas artificialmente
    em células comparadas.
    """

    matched_rows = 0
    compared_rows = 0
    row_metric_cases = 0

    matched_cells = 0
    compared_cells = 0
    cell_metric_cases = 0

    for result in results:
        metadata = result.metadata

        if not isinstance(metadata, Mapping):
            continue

        case_matched_rows = metadata.get("matched_rows")
        case_compared_rows = metadata.get("compared_rows")

        if (
            _is_valid_metric_count(case_matched_rows)
            and _is_valid_metric_count(case_compared_rows)
            and case_matched_rows <= case_compared_rows
        ):
            matched_rows += case_matched_rows
            compared_rows += case_compared_rows
            row_metric_cases += 1

        case_matched_cells = metadata.get("matched_cells")
        case_compared_cells = metadata.get("compared_cells")

        if (
            _is_valid_metric_count(case_matched_cells)
            and _is_valid_metric_count(case_compared_cells)
            and case_matched_cells <= case_compared_cells
        ):
            matched_cells += case_matched_cells
            compared_cells += case_compared_cells
            cell_metric_cases += 1

    row_accuracy = (
        round(matched_rows / compared_rows * 100, 2) if compared_rows else None
    )

    cell_accuracy = (
        round(matched_cells / compared_cells * 100, 2) if compared_cells else None
    )

    return {
        "matched_rows": matched_rows,
        "compared_rows": compared_rows,
        "row_metric_cases": row_metric_cases,
        "row_accuracy": row_accuracy,
        "matched_cells": matched_cells,
        "compared_cells": compared_cells,
        "cell_metric_cases": cell_metric_cases,
        "cell_accuracy": cell_accuracy,
        "cell_accuracy_scope": "matched_rows_only",
    }


# ==========================================================
# SNAPSHOTS DE UTILIZAÇÃO DE LLM
# ==========================================================


def _get_llm_usage_snapshot(
    result: EvaluationCaseResult,
) -> Mapping[str, Any] | None:
    """Obtém e valida a estrutura básica das métricas LLM."""

    metadata = result.metadata

    if not isinstance(metadata, Mapping):
        return None

    snapshot = metadata.get("llm_metrics")

    if not isinstance(snapshot, Mapping):
        return None

    required_sections = (
        "requests",
        "tokens",
        "latency",
    )

    if not all(
        isinstance(snapshot.get(section), Mapping) for section in required_sections
    ):
        return None

    return snapshot


# ==========================================================
# PRECIFICAÇÃO POR CENÁRIO, PROVEDOR E MODELO
# ==========================================================


def _summarize_token_costs(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Calcula estimativas financeiras a partir do consumo de tokens.

    Utiliza as tarifas públicas padrão cadastradas em pricing.py.

    Consolida os custos:
    - Por cenário.
    - Por provedor.
    - Por modelo.
    - No conjunto completo de resultados.

    As estimativas representam preços públicos de referência,
    não valores efetivamente faturados pelos provedores.

    Cenários com tokens não reportados ou modelos sem tarifas
    cadastradas são identificados como incompletos.

    O subtotal conhecido permanece disponível mesmo quando
    não é possível calcular o custo total de um grupo.
    """

    by_case: list[dict[str, Any]] = []

    provider_buckets: dict[str, dict[str, Any]] = {}
    model_buckets: dict[str, dict[str, Any]] = {}

    total_known_cost = Decimal("0")

    estimated_cases = 0
    incomplete_cases = 0

    # ------------------------------------------------------
    # INICIALIZAÇÃO DOS AGRUPAMENTOS
    # ------------------------------------------------------

    def get_bucket(
        container: dict[str, dict[str, Any]],
        key: str,
        provider: str,
        model: str | None,
    ) -> dict[str, Any]:
        """Obtém ou inicializa um agrupamento financeiro."""

        if key not in container:
            container[key] = {
                "provider": provider,
                "model": model,
                "known_subtotal": Decimal("0"),
                "estimated_cases": 0,
                "incomplete_cases": 0,
            }

        return container[key]

    # ------------------------------------------------------
    # PROCESSAMENTO INDIVIDUAL DOS CENÁRIOS
    # ------------------------------------------------------

    for result_index, result in enumerate(results):
        snapshot = _get_llm_usage_snapshot(result)

        if snapshot is None:
            continue

        tokens = snapshot["tokens"]
        requests = snapshot["requests"]

        # --------------------------------------------------
        # IDENTIFICAÇÃO DO PROVEDOR E MODELO
        # --------------------------------------------------

        snapshot_provider = snapshot.get("provider")
        snapshot_model = snapshot.get("model")

        provider_value = (
            snapshot_provider
            if isinstance(snapshot_provider, str) and snapshot_provider.strip()
            else result.provider
        )

        model_value = (
            snapshot_model
            if isinstance(snapshot_model, str) and snapshot_model.strip()
            else result.model
        )

        provider = (
            provider_value.strip().lower()
            if isinstance(provider_value, str) and provider_value.strip()
            else "unknown"
        )

        model = (
            model_value.strip()
            if isinstance(model_value, str) and model_value.strip()
            else None
        )

        # --------------------------------------------------
        # VALIDAÇÃO DOS CONTADORES DE TOKENS
        # --------------------------------------------------

        input_tokens = tokens.get("input")
        output_tokens = tokens.get("output")
        thinking_tokens = tokens.get("thinking", 0)
        unreported_tokens = tokens.get("unreported", 0)

        valid_token_counts = (
            _is_valid_metric_count(input_tokens)
            and _is_valid_metric_count(output_tokens)
            and _is_valid_metric_count(thinking_tokens)
        )

        valid_unreported = _is_valid_metric_count(unreported_tokens)

        requests_total = requests.get("total", 0)

        # --------------------------------------------------
        # IDENTIFICAÇÃO DE CENÁRIOS SEM CONSUMO
        # --------------------------------------------------

        no_usage = (
            _is_valid_metric_count(requests_total)
            and requests_total == 0
            and valid_token_counts
            and input_tokens == 0
            and output_tokens == 0
            and thinking_tokens == 0
            and valid_unreported
            and unreported_tokens == 0
        )

        estimate: dict[str, Any] | None = None
        amount: Decimal | None = None
        reason: str | None = None

        # --------------------------------------------------
        # CÁLCULO DO CUSTO
        # --------------------------------------------------

        if no_usage:
            # Nenhuma chamada e nenhum token consumido:
            # o custo conhecido é exatamente zero.
            amount = Decimal("0")

        elif not valid_token_counts:
            reason = "token_usage_unavailable"

        elif model is None:
            reason = "model_unavailable"

        else:
            estimate = estimate_token_cost(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                pricing_tier="standard",
            )

            if estimate is None:
                reason = "model_not_priced"

            else:
                amount = Decimal(str(estimate["estimated_cost_usd"]))

        # --------------------------------------------------
        # COMPLETUDE DA ESTIMATIVA
        # --------------------------------------------------

        incomplete = amount is None or not valid_unreported or unreported_tokens > 0

        if reason is None:
            if not valid_unreported:
                reason = "unreported_token_count_invalid"

            elif unreported_tokens > 0:
                reason = "some_requests_without_token_usage"

        if amount is not None:
            total_known_cost += amount

        if incomplete:
            incomplete_cases += 1
        else:
            estimated_cases += 1

        # --------------------------------------------------
        # RESULTADO INDIVIDUAL
        # --------------------------------------------------

        case_estimate = {
            "case_id": result.case_id,
            "result_index": result_index,
            "provider": provider,
            "model": model,
            "estimated_cost_usd": (
                _round_cost(amount) if amount is not None and not incomplete else None
            ),
            "known_subtotal_usd": (_round_cost(amount) if amount is not None else 0.0),
            "status": "incomplete" if incomplete else "estimated",
            "reason": reason,
            "pricing": estimate,
        }

        # Lista preserva cenários com IDs repetidos,
        # inclusive quando houver futuras execuções repetidas.
        by_case.append(case_estimate)

        # --------------------------------------------------
        # AGRUPAMENTO POR PROVEDOR
        # --------------------------------------------------

        provider_bucket = get_bucket(
            provider_buckets,
            provider,
            provider,
            None,
        )

        # --------------------------------------------------
        # AGRUPAMENTO POR MODELO
        # --------------------------------------------------

        model_key = f"{provider}/{model or 'unknown'}"

        model_bucket = get_bucket(
            model_buckets,
            model_key,
            provider,
            model,
        )

        # --------------------------------------------------
        # ATUALIZAÇÃO DOS AGRUPAMENTOS
        # --------------------------------------------------

        for bucket in (provider_bucket, model_bucket):
            if amount is not None:
                bucket["known_subtotal"] += amount

            if incomplete:
                bucket["incomplete_cases"] += 1
            else:
                bucket["estimated_cases"] += 1

    # ------------------------------------------------------
    # SERIALIZAÇÃO DOS AGRUPAMENTOS
    # ------------------------------------------------------

    def serialize_bucket(
        bucket: dict[str, Any],
    ) -> dict[str, Any]:
        """Converte um agrupamento financeiro para dicionário."""

        incomplete = bucket["incomplete_cases"] > 0
        subtotal = bucket["known_subtotal"]

        return {
            "provider": bucket["provider"],
            "model": bucket["model"],
            "estimated_cost_usd": (_round_cost(subtotal) if not incomplete else None),
            "known_subtotal_usd": _round_cost(subtotal),
            "estimated_cases": bucket["estimated_cases"],
            "incomplete_cases": bucket["incomplete_cases"],
        }

    # ------------------------------------------------------
    # CONSOLIDAÇÃO GERAL
    # ------------------------------------------------------

    complete_estimate = bool(by_case) and incomplete_cases == 0

    return {
        "basis": "standard_public_list_price",
        "currency": "USD",
        "not_actual_invoice": True,
        "observed_cases": len(by_case),
        "estimated_cost_usd": (
            _round_cost(total_known_cost) if complete_estimate else None
        ),
        "known_subtotal_usd": _round_cost(total_known_cost),
        "estimated_cases": estimated_cases,
        "incomplete_cases": incomplete_cases,
        "by_case": by_case,
        "by_provider": {
            key: serialize_bucket(value)
            for key, value in sorted(provider_buckets.items())
        },
        "by_model": {
            key: serialize_bucket(value) for key, value in sorted(model_buckets.items())
        },
    }


# ==========================================================
# CONSOLIDAÇÃO DE UTILIZAÇÃO DE LLM
# ==========================================================


def _summarize_llm_usage(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Consolida o consumo de LLM dos cenários avaliados.

    Agrega requisições, tokens, retries e latência a partir
    dos snapshots individuais armazenados nos metadados.

    Os campos históricos de precificação permanecem compatíveis
    com a implementação anterior.

    A nova estrutura token_cost_estimate calcula custos usando
    tarifas públicas padrão por modelo e consumo de tokens.

    Contadores de requisições inconsistentes não participam das
    taxas de sucesso e falha. A quantidade total observada continua
    disponível, mas a taxa usa somente snapshots com total,
    sucessos e falhas coerentes.

    A média de latência por requisição utiliza apenas os snapshots
    que possuem, simultaneamente, latência e quantidade de
    requisições válidas.
    """

    observed_cases = 0
    cases_with_requests = 0
    cases_without_requests = 0
    cases_with_unknown_request_count = 0

    request_metric_cases = 0
    request_outcome_invalid_cases = 0
    request_retry_invalid_cases = 0

    requests_total = 0
    requests_with_valid_outcomes = 0
    requests_successful = 0
    requests_failed = 0
    retries = 0

    token_totals = {
        "input": 0,
        "output": 0,
        "thinking": 0,
        "total": 0,
        "unreported": 0,
    }

    total_latency_ms = 0.0
    latency_cases: list[float] = []

    latency_with_request_count_ms = 0.0
    latency_request_count = 0

    providers: set[str] = set()

    for result in results:
        snapshot = _get_llm_usage_snapshot(result)

        if snapshot is None:
            continue

        observed_cases += 1

        snapshot_provider = snapshot.get("provider")

        if isinstance(snapshot_provider, str) and snapshot_provider.strip():
            provider = snapshot_provider.strip().lower()

        elif isinstance(result.provider, str) and result.provider.strip():
            provider = result.provider.strip().lower()

        else:
            provider = "unknown"

        providers.add(provider)

        requests = snapshot["requests"]
        tokens = snapshot["tokens"]
        latency = snapshot["latency"]

        # --------------------------------------------------
        # REQUISIÇÕES
        # --------------------------------------------------

        case_requests_total = requests.get("total")
        case_requests_successful = requests.get("successful")
        case_requests_failed = requests.get("failed")
        case_retries = requests.get("retries")

        valid_request_total = _is_valid_metric_count(case_requests_total)

        if valid_request_total:
            request_metric_cases += 1
            requests_total += case_requests_total

            if case_requests_total > 0:
                cases_with_requests += 1
            else:
                cases_without_requests += 1

        else:
            cases_with_unknown_request_count += 1

        valid_request_outcomes = (
            valid_request_total
            and _is_valid_metric_count(case_requests_successful)
            and _is_valid_metric_count(case_requests_failed)
            and (case_requests_successful + case_requests_failed <= case_requests_total)
        )

        if valid_request_outcomes:
            requests_with_valid_outcomes += case_requests_total
            requests_successful += case_requests_successful
            requests_failed += case_requests_failed

        else:
            request_outcome_invalid_cases += 1

        if _is_valid_metric_count(case_retries):
            retries += case_retries
        else:
            request_retry_invalid_cases += 1

        # --------------------------------------------------
        # TOKENS
        # --------------------------------------------------

        # Mantém compatibilidade com os nomes utilizados
        # pelo snapshot de AnalysisLLMMetrics.

        for token_field in token_totals:
            value = tokens.get(token_field, 0)

            if _is_valid_metric_count(value):
                token_totals[token_field] += value

        # --------------------------------------------------
        # LATÊNCIA
        # --------------------------------------------------

        case_latency = latency.get("total_ms")

        if _is_valid_metric_number(case_latency):
            normalized_latency = float(case_latency)

            total_latency_ms += normalized_latency
            latency_cases.append(normalized_latency)

            # A média por requisição só pode usar como
            # denominador requisições do mesmo snapshot
            # que forneceu uma latência válida.
            if valid_request_total and case_requests_total > 0:
                latency_with_request_count_ms += normalized_latency
                latency_request_count += case_requests_total

    # ------------------------------------------------------
    # TAXAS DE REQUISIÇÕES
    # ------------------------------------------------------

    request_success_rate = (
        round(
            requests_successful / requests_with_valid_outcomes * 100,
            2,
        )
        if requests_with_valid_outcomes
        else None
    )

    request_failure_rate = (
        round(
            requests_failed / requests_with_valid_outcomes * 100,
            2,
        )
        if requests_with_valid_outcomes
        else None
    )

    # ------------------------------------------------------
    # LATÊNCIA MÉDIA
    # ------------------------------------------------------

    average_request_latency_ms = (
        round(
            latency_with_request_count_ms / latency_request_count,
            2,
        )
        if latency_request_count
        else None
    )

    average_case_latency_ms = round(fmean(latency_cases), 2) if latency_cases else None

    # ------------------------------------------------------
    # METADADOS LEGADOS DE COBRANÇA
    # ------------------------------------------------------

    pricing_by_provider: dict[str, dict[str, Any]] = {}

    for provider in sorted(providers):
        if provider not in PROVIDER_MODELS:
            pricing_by_provider[provider] = {
                "pricing_tier": "unknown",
                "cost_estimate_usd": None,
            }

            continue

        pricing_by_provider[provider] = get_pricing_metadata(provider)

    pricing_tiers = {
        metadata.get("pricing_tier", "unknown")
        for metadata in pricing_by_provider.values()
    }

    if not pricing_tiers:
        pricing_tier = "unknown"

    elif len(pricing_tiers) == 1:
        pricing_tier = next(iter(pricing_tiers))

    else:
        pricing_tier = "mixed"

    provider_costs = [
        metadata.get("cost_estimate_usd") for metadata in pricing_by_provider.values()
    ]

    if not provider_costs or any(cost is None for cost in provider_costs):
        cost_estimate_usd = None

    elif all(_is_valid_metric_number(cost) for cost in provider_costs):
        # Compatibilidade com o contrato anterior:
        # os metadados são contabilizados uma vez por provedor.
        # Não representam custos calculados por token.
        cost_estimate_usd = round(sum(provider_costs), 8)

    else:
        cost_estimate_usd = None

    # ------------------------------------------------------
    # NOVA ESTIMATIVA FINANCEIRA POR CONSUMO
    # ------------------------------------------------------

    token_cost_estimate = _summarize_token_costs(results)

    # ------------------------------------------------------
    # RETORNO CONSOLIDADO
    # ------------------------------------------------------

    return {
        "observed_cases": observed_cases,
        "cases_with_requests": cases_with_requests,
        "cases_without_requests": cases_without_requests,
        "cases_with_unknown_request_count": cases_with_unknown_request_count,
        "request_metric_cases": request_metric_cases,
        "request_outcome_invalid_cases": request_outcome_invalid_cases,
        "request_retry_invalid_cases": request_retry_invalid_cases,
        "providers": sorted(providers),
        "requests": {
            "total": requests_total,
            "successful": requests_successful,
            "failed": requests_failed,
            "retries": retries,
            "success_rate": request_success_rate,
            "failure_rate": request_failure_rate,
        },
        "tokens": token_totals,
        "latency": {
            "total_ms": round(total_latency_ms, 2),
            "average_request_ms": average_request_latency_ms,
            "average_case_ms": average_case_latency_ms,
            "measured_cases": len(latency_cases),
        },
        "pricing_tier": pricing_tier,
        "cost_estimate_usd": cost_estimate_usd,
        "pricing_by_provider": pricing_by_provider,
        "token_cost_estimate": token_cost_estimate,
    }


# ==========================================================
# MÉTRICAS GERAIS
# ==========================================================


def _summarize(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Calcula métricas de um conjunto de resultados."""

    total_cases = len(results)

    passed = sum(result.status == "passed" for result in results)

    failed = sum(result.status == "failed" for result in results)

    errors = sum(result.status == "error" for result in results)

    skipped = sum(result.status == "skipped" for result in results)

    evaluated = passed + failed + errors

    scored_results = [
        result for result in results if result.status in ("passed", "failed")
    ]

    durations = [
        result.duration_ms for result in results if result.duration_ms is not None
    ]

    cache_results = [result for result in results if result.cache_hit is not None]

    cache_hits = sum(result.cache_hit is True for result in cache_results)

    cache_misses = sum(result.cache_hit is False for result in cache_results)

    return {
        "total_cases": total_cases,
        "evaluated_cases": evaluated,
        "passed_cases": passed,
        "failed_cases": failed,
        "error_cases": errors,
        "skipped_cases": skipped,
        "pass_rate": round(
            passed / evaluated * 100,
            2,
        )
        if evaluated
        else 0.0,
        "scored_cases": len(scored_results),
        "average_score": round(
            fmean(result.score for result in scored_results),
            4,
        )
        if scored_results
        else 0.0,
        "average_duration_ms": round(
            fmean(durations),
            2,
        )
        if durations
        else None,
        "cache_observations": len(cache_results),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": round(
            cache_hits / len(cache_results) * 100,
            2,
        )
        if cache_results
        else None,
        "analysis_mode_adherence": _summarize_analysis_mode_adherence(results),
        **_summarize_result_quality(results),
        "llm_usage": _summarize_llm_usage(results),
    }


# ==========================================================
# MÉTRICAS SEGMENTADAS
# ==========================================================


def calculate_evaluation_metrics(
    results: Iterable[EvaluationCaseResult],
) -> dict[str, Any]:
    """Calcula métricas gerais e segmentadas de uma avaliação.

    Args:
        results: Resultados individuais dos cenários avaliados.

    Returns:
        Dicionário com métricas gerais, por categoria,
        por provedor e por combinação de provedor/modelo.
    """

    normalized_results = list(results)

    by_category: dict[
        str,
        list[EvaluationCaseResult],
    ] = defaultdict(list)

    by_provider: dict[
        str,
        list[EvaluationCaseResult],
    ] = defaultdict(list)

    by_model: dict[
        str,
        list[EvaluationCaseResult],
    ] = defaultdict(list)

    for result in normalized_results:
        metadata = result.metadata

        category = str(
            metadata.get(
                "category",
                "uncategorized",
            )
            if isinstance(metadata, Mapping)
            else "uncategorized"
        )

        provider = result.provider or "unknown"

        model = result.model or "unknown"

        by_category[category].append(result)

        by_provider[provider].append(result)

        model_key = f"{provider}/{model}"

        by_model[model_key].append(result)

    return {
        "summary": _summarize(normalized_results),
        "by_category": {
            category: _summarize(category_results)
            for category, category_results in sorted(by_category.items())
        },
        "by_provider": {
            provider: _summarize(provider_results)
            for provider, provider_results in sorted(by_provider.items())
        },
        "by_model": {
            model: _summarize(model_results)
            for model, model_results in sorted(by_model.items())
        },
    }
