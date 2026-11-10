"""Métricas consolidadas do AI Evaluation Engine."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any

from ai_analyst.evaluation.models import EvaluationCaseResult


def _summarize_analysis_mode_adherence(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Calcula a aderência aos modos de análise esperados.

    A avaliação considera somente cenários que definiram
    expected_analysis_mode.

    Um modo é considerado observado quando o resultado contém
    um valor de analysis_mode não vazio do tipo string.

    Cenários sem modo observado são contabilizados como
    unobserved_cases e não entram no denominador da taxa.

    A aderência é independente do status final do cenário.
    Por exemplo, um cenário pode ter encontrado o modo esperado,
    mas posteriormente apresentar um erro operacional.

    Returns:
        Métricas de aderência ao modo de análise.
    """

    expected_cases = 0
    observed_cases = 0
    matches = 0
    mismatches = 0

    for result in results:
        metadata = result.metadata

        expected_mode = metadata.get("expected_analysis_mode")

        # Cenários sem expectativa explícita não participam
        # das métricas de aderência.
        if expected_mode is None:
            continue

        expected_cases += 1

        actual_mode = metadata.get("analysis_mode")

        # Erros anteriores à execução do agente podem não
        # possuir um modo de análise observável.
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
        "analysis_mode_adherence": (_summarize_analysis_mode_adherence(results)),
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
