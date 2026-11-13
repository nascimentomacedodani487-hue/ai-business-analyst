"""Métricas consolidadas do AI Evaluation Engine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import isfinite
from statistics import fmean
from typing import Any

from ai_analyst.evaluation.models import EvaluationCaseResult
from ai_analyst.llm.metrics import PROVIDER_MODELS, get_pricing_metadata


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


def _is_valid_metric_count(value: Any) -> bool:
    """Verifica se um contador é um inteiro não negativo válido."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_valid_metric_number(value: Any) -> bool:
    """Verifica se uma métrica numérica é finita e não negativa."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )


def _summarize_result_quality(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Consolida as métricas de correspondência de linhas e células."""

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
    }


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


def _summarize_llm_usage(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Consolida o consumo de LLM dos cenários avaliados.

    Agrega requisições, tokens, retries e latência a partir
    dos snapshots individuais armazenados nos metadados.

    O custo utiliza somente as informações de cobrança disponíveis
    em get_pricing_metadata(), sem inventar tarifas por token.
    """

    observed_cases = 0
    cases_with_requests = 0

    requests_total = 0
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

        # ------------------------------------------------------
        # REQUISIÇÕES
        # ------------------------------------------------------

        case_requests_total = requests.get("total", 0)
        case_requests_successful = requests.get("successful", 0)
        case_requests_failed = requests.get("failed", 0)
        case_retries = requests.get("retries", 0)

        if _is_valid_metric_count(case_requests_total):
            requests_total += case_requests_total

            if case_requests_total > 0:
                cases_with_requests += 1

        if _is_valid_metric_count(case_requests_successful):
            requests_successful += case_requests_successful

        if _is_valid_metric_count(case_requests_failed):
            requests_failed += case_requests_failed

        if _is_valid_metric_count(case_retries):
            retries += case_retries

        # ------------------------------------------------------
        # TOKENS
        # ------------------------------------------------------

        # Os nomes das propriedades são os mesmos utilizados
        # pelo snapshot de AnalysisLLMMetrics.
        #
        # Isso evita divergências entre nomes de origem
        # e nomes de destino durante a consolidação.

        for token_field in token_totals:
            value = tokens.get(token_field, 0)

            if _is_valid_metric_count(value):
                token_totals[token_field] += value

        # ------------------------------------------------------
        # LATÊNCIA
        # ------------------------------------------------------

        case_latency = latency.get("total_ms")

        if _is_valid_metric_number(case_latency):
            normalized_latency = float(case_latency)

            total_latency_ms += normalized_latency
            latency_cases.append(normalized_latency)

    request_success_rate = (
        round(requests_successful / requests_total * 100, 2) if requests_total else None
    )

    request_failure_rate = (
        round(requests_failed / requests_total * 100, 2) if requests_total else None
    )

    average_request_latency_ms = (
        round(total_latency_ms / requests_total, 2) if requests_total else None
    )

    average_case_latency_ms = round(fmean(latency_cases), 2) if latency_cases else None

    # ----------------------------------------------------------
    # METADADOS DE COBRANÇA
    # ----------------------------------------------------------

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
        # Os metadados são contabilizados uma vez por provedor.
        # Não representam preços unitários por chamada ou token.
        cost_estimate_usd = round(sum(provider_costs), 8)
    else:
        cost_estimate_usd = None

    return {
        "observed_cases": observed_cases,
        "cases_with_requests": cases_with_requests,
        "cases_without_requests": observed_cases - cases_with_requests,
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
    }


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
        category = str(
            result.metadata.get(
                "category",
                "uncategorized",
            )
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
