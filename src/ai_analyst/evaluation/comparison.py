"""Comparação histórica de benchmarks do AI Evaluation Engine.

Compara dois relatórios consolidados do BenchmarkRunner,
identificando regressões, melhorias e alterações de desempenho.

A comparação exige compatibilidade entre os datasets e considera
somente pares com o mesmo provedor e modelo.

As métricas financeiras são deliberadamente excluídas:
os metadados de cobrança disponíveis não representam necessariamente
o custo real de utilização de tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from typing import Any

# ==========================================================
# CONFIGURAÇÃO
# ==========================================================


DEFAULT_THRESHOLDS: dict[str, float] = {
    "pass_rate": 0.5,
    "average_score": 0.005,
    "row_accuracy": 0.5,
    "cell_accuracy": 0.5,
    "average_duration_ms": 10.0,
    "llm_usage.tokens.input": 10.0,
    "llm_usage.tokens.output": 10.0,
    "llm_usage.tokens.thinking": 10.0,
    "llm_usage.tokens.total": 10.0,
    "llm_usage.tokens.unreported": 10.0,
    "llm_usage.latency.average_request_ms": 10.0,
    "error_cases": 0.0,
    "failed_cases": 0.0,
    "skipped_cases": 0.0,
    "llm_usage.requests.failed": 0.0,
    "llm_usage.requests.retries": 0.0,
}


METRIC_DIRECTIONS: dict[str, str] = {
    "pass_rate": "higher",
    "average_score": "higher",
    "row_accuracy": "higher",
    "cell_accuracy": "higher",
    "average_duration_ms": "lower",
    "error_cases": "lower",
    "failed_cases": "lower",
    "skipped_cases": "lower",
    "llm_usage.tokens.input": "lower",
    "llm_usage.tokens.output": "lower",
    "llm_usage.tokens.thinking": "lower",
    "llm_usage.tokens.total": "lower",
    "llm_usage.tokens.unreported": "lower",
    "llm_usage.requests.failed": "lower",
    "llm_usage.requests.retries": "lower",
    "llm_usage.latency.average_request_ms": "lower",
}


# Métricas comparadas por diferença absoluta.
ABSOLUTE_THRESHOLD_METRICS = {
    "pass_rate",
    "average_score",
    "row_accuracy",
    "cell_accuracy",
    "error_cases",
    "failed_cases",
    "skipped_cases",
    "llm_usage.requests.failed",
    "llm_usage.requests.retries",
}


INTEGER_METRICS = {
    "error_cases",
    "failed_cases",
    "skipped_cases",
    "llm_usage.tokens.input",
    "llm_usage.tokens.output",
    "llm_usage.tokens.thinking",
    "llm_usage.tokens.total",
    "llm_usage.tokens.unreported",
    "llm_usage.requests.failed",
    "llm_usage.requests.retries",
}


BOUNDED_METRICS: dict[str, tuple[float, float]] = {
    "pass_rate": (0.0, 100.0),
    "average_score": (0.0, 1.0),
    "row_accuracy": (0.0, 100.0),
    "cell_accuracy": (0.0, 100.0),
}


# ==========================================================
# UTILITÁRIOS
# ==========================================================


def _is_valid_number(value: Any) -> bool:
    """Verifica se um valor numérico é finito e não negativo."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
    )


def _is_valid_integer(value: Any) -> bool:
    """Verifica se um contador é um inteiro não negativo."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _get_nested_value(
    data: Mapping[str, Any] | None,
    path: str,
) -> Any:
    """Recupera um valor usando um caminho separado por pontos."""

    if not isinstance(data, Mapping):
        return None

    current: Any = data

    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None

        current = current.get(part)

    return current


def _round_number(value: float | int) -> float | int:
    """Arredonda resultados numéricos sem perder precisão desnecessariamente."""

    rounded = round(value, 4)

    if isinstance(value, int) and rounded == value:
        return value

    return rounded


def _normalize_model(value: Any) -> str | None:
    """Normaliza um identificador de modelo."""

    if not isinstance(value, str) or not value.strip():
        return None

    return value.strip()


def _normalize_provider(value: Any) -> str | None:
    """Normaliza um identificador de provedor."""

    if not isinstance(value, str) or not value.strip():
        return None

    return value.strip().lower()


def _exceeds_threshold(
    difference: float | int,
    threshold: float,
) -> bool:
    """Verifica se uma diferença ultrapassa a tolerância.

    isclose evita classificar como regressão uma diferença que,
    devido à representação binária de floats, fica ligeiramente
    acima do limite configurado.
    """

    magnitude = abs(difference)

    if magnitude <= threshold:
        return False

    return not isclose(
        magnitude,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


# ==========================================================
# VALIDAÇÃO DOS RELATÓRIOS
# ==========================================================


def _require_report(
    report: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Valida a estrutura básica de um relatório."""

    if not isinstance(report, Mapping):
        raise TypeError(f"{field_name} deve ser um objeto de mapeamento.")

    benchmark_id = report.get("benchmark_id")

    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError(f"{field_name}.benchmark_id deve ser uma string não vazia.")

    return report


def _require_dataset(
    report: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Valida a presença da estrutura do dataset."""

    dataset = report.get("dataset")

    if not isinstance(dataset, Mapping):
        raise ValueError(f"{field_name}.dataset deve ser um objeto de mapeamento.")

    fingerprint = dataset.get("sha256")

    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint.lower())
    ):
        raise ValueError(
            f"{field_name}.dataset.sha256 deve ser um hash SHA-256 hexadecimal."
        )

    case_ids = dataset.get("case_ids")

    if (
        not isinstance(case_ids, list)
        or not case_ids
        or not all(isinstance(case_id, str) and case_id.strip() for case_id in case_ids)
    ):
        raise ValueError(
            f"{field_name}.dataset.case_ids deve conter identificadores válidos."
        )

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{field_name}.dataset.case_ids não pode conter duplicatas.")

    return dataset


def _validate_optional_report_sections(
    report: Mapping[str, Any],
    field_name: str,
) -> None:
    """Valida as estruturas utilizadas na comparação."""

    for section in ("configuration", "provider_runs", "comparison"):
        value = report.get(section)

        if value is not None and not isinstance(value, Mapping):
            raise ValueError(
                f"{field_name}.{section} deve ser um objeto de mapeamento."
            )

    comparison = report.get("comparison", {})

    if isinstance(comparison, Mapping):
        by_provider = comparison.get("by_provider")

        if by_provider is not None and not isinstance(by_provider, Mapping):
            raise ValueError(
                f"{field_name}.comparison.by_provider deve ser um mapeamento."
            )


# ==========================================================
# COMPATIBILIDADE
# ==========================================================


def _get_dataset_compatibility(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Verifica se dois relatórios utilizam datasets compatíveis."""

    baseline_dataset = _require_dataset(
        baseline,
        "baseline",
    )

    candidate_dataset = _require_dataset(
        candidate,
        "candidate",
    )

    reasons: list[str] = []

    baseline_fingerprint = baseline_dataset["sha256"].lower()

    candidate_fingerprint = candidate_dataset["sha256"].lower()

    if baseline_fingerprint != candidate_fingerprint:
        reasons.append("dataset_fingerprint_mismatch")

    baseline_cases = set(baseline_dataset["case_ids"])

    candidate_cases = set(candidate_dataset["case_ids"])

    if baseline_cases != candidate_cases:
        reasons.append("dataset_case_ids_mismatch")

    baseline_count = baseline_dataset.get("case_count")

    candidate_count = candidate_dataset.get("case_count")

    if (
        not isinstance(baseline_count, int)
        or isinstance(baseline_count, bool)
        or baseline_count != len(baseline_cases)
    ):
        reasons.append("baseline_dataset_case_count_invalid")

    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count != len(candidate_cases)
    ):
        reasons.append("candidate_dataset_case_count_invalid")

    if (
        isinstance(baseline_count, int)
        and not isinstance(baseline_count, bool)
        and isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and baseline_count != candidate_count
    ):
        reasons.append("dataset_case_count_mismatch")

    baseline_version = baseline_dataset.get("version")

    candidate_version = candidate_dataset.get("version")

    if baseline_version != candidate_version:
        reasons.append("dataset_version_mismatch")

    return {
        "comparable": not reasons,
        "reasons": reasons,
        "baseline_fingerprint": baseline_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "case_count": len(baseline_cases),
        "same_case_ids": baseline_cases == candidate_cases,
    }


# ==========================================================
# IDENTIFICAÇÃO DE PROVEDORES
# ==========================================================


def _get_provider_names(
    report: Mapping[str, Any],
) -> set[str]:
    """Obtém os provedores identificados no relatório."""

    providers: set[str] = set()

    configuration = report.get("configuration", {})

    if isinstance(configuration, Mapping):
        configured_providers = configuration.get("providers", [])

        if configured_providers is not None:
            if not isinstance(configured_providers, list):
                raise ValueError("configuration.providers deve ser uma lista.")

            for provider in configured_providers:
                normalized = _normalize_provider(provider)

                if normalized is None:
                    raise ValueError(
                        "configuration.providers contém um provedor inválido."
                    )

                providers.add(normalized)

    provider_runs = report.get("provider_runs", {})

    if isinstance(provider_runs, Mapping):
        for provider in provider_runs:
            normalized = _normalize_provider(provider)

            if normalized is not None:
                providers.add(normalized)

    comparison = report.get("comparison", {})

    if isinstance(comparison, Mapping):
        by_provider = comparison.get("by_provider", {})

        if isinstance(by_provider, Mapping):
            for provider in by_provider:
                normalized = _normalize_provider(provider)

                if normalized is not None:
                    providers.add(normalized)

    return providers


def _get_provider_information(
    report: Mapping[str, Any],
    provider: str,
) -> dict[str, Any]:
    """Recupera estado, modelo e métricas de um provedor."""

    provider_runs = report.get("provider_runs", {})

    comparison = report.get("comparison", {})

    if not isinstance(provider_runs, Mapping):
        provider_runs = {}

    if not isinstance(comparison, Mapping):
        comparison = {}

    by_provider = comparison.get("by_provider", {})

    if not isinstance(by_provider, Mapping):
        by_provider = {}

    execution = provider_runs.get(provider, {})

    provider_comparison = by_provider.get(provider, {})

    if not isinstance(execution, Mapping):
        execution = {}

    if not isinstance(provider_comparison, Mapping):
        provider_comparison = {}

    individual_report = execution.get("report")

    if not isinstance(individual_report, Mapping):
        individual_report = {}

    execution_status = execution.get("execution_status")

    if execution_status not in {"completed", "error"}:
        execution_status = provider_comparison.get("execution_status")

    if execution_status not in {"completed", "error"}:
        execution_status = None

    model = _normalize_model(provider_comparison.get("model"))

    if model is None:
        model = _normalize_model(individual_report.get("model"))

    metrics = provider_comparison.get("metrics")

    if not isinstance(metrics, Mapping):
        metrics = None

    return {
        "execution_status": execution_status,
        "model": model,
        "metrics": metrics,
    }


# ==========================================================
# DEFINIÇÃO E VALIDAÇÃO DAS MÉTRICAS
# ==========================================================


def _get_metric_value(
    metrics: Mapping[str, Any] | None,
    metric_name: str,
) -> Any:
    """Obtém uma métrica consolidada de um provedor."""

    return _get_nested_value(
        metrics,
        metric_name,
    )


def _validate_metric_value(
    metric_name: str,
    value: Any,
) -> bool:
    """Valida uma métrica de acordo com sua natureza."""

    if value is None:
        return False

    if metric_name in INTEGER_METRICS:
        return _is_valid_integer(value)

    if not _is_valid_number(value):
        return False

    bounds = BOUNDED_METRICS.get(metric_name)

    if bounds is not None:
        minimum, maximum = bounds

        return minimum <= value <= maximum

    return True


# ==========================================================
# COMPARAÇÃO NUMÉRICA
# ==========================================================


def _compare_metric(
    metric_name: str,
    baseline_value: Any,
    candidate_value: Any,
    threshold: float,
) -> dict[str, Any]:
    """Compara dois valores numéricos de uma mesma métrica."""

    if not _validate_metric_value(
        metric_name, baseline_value
    ) or not _validate_metric_value(metric_name, candidate_value):
        return {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": None,
            "delta_pct": None,
            "threshold": threshold,
            "status": "not_comparable",
        }

    delta = candidate_value - baseline_value

    raw_delta_pct = delta / abs(baseline_value) * 100 if baseline_value != 0 else None

    delta_pct = _round_number(raw_delta_pct) if raw_delta_pct is not None else None

    rounded_delta = _round_number(delta)

    if delta == 0:
        status = "stable"

    else:
        if metric_name in ABSOLUTE_THRESHOLD_METRICS:
            threshold_is_exceeded = _exceeds_threshold(
                delta,
                threshold,
            )

        elif raw_delta_pct is not None:
            threshold_is_exceeded = _exceeds_threshold(
                raw_delta_pct,
                threshold,
            )

        else:
            # Quando a referência é zero, não existe uma base percentual.
            # Qualquer alteração não nula é considerada relevante.
            threshold_is_exceeded = True

        if not threshold_is_exceeded:
            status = "stable"

        elif METRIC_DIRECTIONS[metric_name] == "higher":
            status = "improvement" if delta > 0 else "regression"

        else:
            status = "improvement" if delta < 0 else "regression"

    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": rounded_delta,
        "delta_pct": delta_pct,
        "threshold": threshold,
        "status": status,
    }


# ==========================================================
# CLASSIFICAÇÃO CONSOLIDADA
# ==========================================================


def _combine_statuses(
    statuses: list[str],
    *,
    empty_status: str = "not_comparable",
) -> str:
    """Combina estados individuais em uma classificação consolidada."""

    has_regression = "regression" in statuses

    has_improvement = "improvement" in statuses

    if has_regression and has_improvement:
        return "mixed"

    if has_regression:
        return "regression"

    if has_improvement:
        return "improvement"

    comparable_statuses = {
        "stable",
        "regression",
        "improvement",
    }

    if any(status in comparable_statuses for status in statuses):
        return "stable"

    return empty_status


# ==========================================================
# COMPARADOR
# ==========================================================


class BenchmarkComparator:
    """Compara dois benchmarks históricos.

    A comparação só é realizada quando os fingerprints e os
    identificadores dos cenários são compatíveis.

    Métricas de provedores só são comparadas quando o provedor
    e o modelo são os mesmos nas duas execuções.

    Args:
        thresholds: tolerâncias opcionais por métrica.

            Para métricas de qualidade, as tolerâncias são absolutas.

            Para métricas de desempenho e consumo, as tolerâncias
            são relativas, expressas em porcentagem.

            Contadores de erros, falhas e retries utilizam
            tolerância absoluta.
    """

    def __init__(
        self,
        thresholds: Mapping[str, float] | None = None,
    ) -> None:
        self.thresholds = dict(DEFAULT_THRESHOLDS)

        if thresholds is None:
            return

        if not isinstance(thresholds, Mapping):
            raise TypeError("thresholds deve ser um objeto de mapeamento.")

        for metric_name, threshold in thresholds.items():
            if metric_name not in METRIC_DIRECTIONS:
                raise ValueError(
                    f"Métrica sem suporte para configuração de tolerância: "
                    f"{metric_name}."
                )

            if not _is_valid_number(threshold):
                raise ValueError(
                    f"A tolerância de '{metric_name}' deve ser "
                    "um número finito e não negativo."
                )

            self.thresholds[metric_name] = float(threshold)

    # ======================================================
    # MÉTRICAS DE UM PROVEDOR
    # ======================================================

    def _compare_provider_metrics(
        self,
        baseline_metrics: Mapping[str, Any] | None,
        candidate_metrics: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Compara todas as métricas suportadas de um provedor."""

        comparisons: dict[str, dict[str, Any]] = {}

        for metric_name in METRIC_DIRECTIONS:
            baseline_value = _get_metric_value(
                baseline_metrics,
                metric_name,
            )

            candidate_value = _get_metric_value(
                candidate_metrics,
                metric_name,
            )

            comparisons[metric_name] = _compare_metric(
                metric_name=metric_name,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                threshold=self.thresholds[metric_name],
            )

        return comparisons

    # ======================================================
    # EXECUÇÃO DO PROVEDOR
    # ======================================================

    @staticmethod
    def _compare_execution(
        baseline_status: str | None,
        candidate_status: str | None,
    ) -> dict[str, Any]:
        """Compara os estados de execução do provedor."""

        if baseline_status == "completed" and candidate_status == "completed":
            status = "stable"

        elif baseline_status == "completed" and candidate_status != "completed":
            status = "regression" if candidate_status == "error" else "not_comparable"

        elif baseline_status != "completed" and candidate_status == "completed":
            status = "improvement" if baseline_status == "error" else "not_comparable"

        elif baseline_status == candidate_status == "error":
            status = "stable"

        else:
            status = "not_comparable"

        return {
            "baseline": baseline_status,
            "candidate": candidate_status,
            "status": status,
        }

    # ======================================================
    # COMPARAÇÃO INDIVIDUAL
    # ======================================================

    def _compare_provider(
        self,
        provider: str,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        baseline_exists: bool,
        candidate_exists: bool,
    ) -> dict[str, Any]:
        """Compara um provedor entre dois benchmarks."""

        if not baseline_exists:
            candidate_info = _get_provider_information(
                candidate,
                provider,
            )

            return {
                "status": "candidate_only",
                "baseline_model": None,
                "candidate_model": candidate_info["model"],
                "execution": {
                    "baseline": None,
                    "candidate": candidate_info["execution_status"],
                    "status": "not_comparable",
                },
                "metrics": {},
            }

        if not candidate_exists:
            baseline_info = _get_provider_information(
                baseline,
                provider,
            )

            return {
                "status": "baseline_only",
                "baseline_model": baseline_info["model"],
                "candidate_model": None,
                "execution": {
                    "baseline": baseline_info["execution_status"],
                    "candidate": None,
                    "status": "not_comparable",
                },
                "metrics": {},
            }

        baseline_info = _get_provider_information(
            baseline,
            provider,
        )

        candidate_info = _get_provider_information(
            candidate,
            provider,
        )

        baseline_status = baseline_info["execution_status"]

        candidate_status = candidate_info["execution_status"]

        execution_comparison = self._compare_execution(
            baseline_status,
            candidate_status,
        )

        baseline_model = baseline_info["model"]

        candidate_model = candidate_info["model"]

        if execution_comparison["status"] in {"regression", "improvement"}:
            return {
                "status": execution_comparison["status"],
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "execution": execution_comparison,
                "metrics": {},
            }

        if baseline_status != "completed" or candidate_status != "completed":
            return {
                "status": "not_comparable",
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "execution": execution_comparison,
                "metrics": {},
            }

        if baseline_model is None or candidate_model is None:
            return {
                "status": "model_unknown",
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "execution": execution_comparison,
                "metrics": {},
            }

        if baseline_model != candidate_model:
            return {
                "status": "model_changed",
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "execution": execution_comparison,
                "metrics": {},
            }

        metrics = self._compare_provider_metrics(
            baseline_metrics=baseline_info["metrics"],
            candidate_metrics=candidate_info["metrics"],
        )

        metric_statuses = [metric["status"] for metric in metrics.values()]

        comparison_status = _combine_statuses(
            metric_statuses,
            empty_status="not_comparable",
        )

        if comparison_status == "stable":
            provider_status = "compared"
        else:
            provider_status = comparison_status

        return {
            "status": provider_status,
            "baseline_model": baseline_model,
            "candidate_model": candidate_model,
            "execution": execution_comparison,
            "metrics": metrics,
        }

    # ======================================================
    # MÉTRICAS CONSOLIDADAS
    # ======================================================

    @staticmethod
    def _summarize_provider_results(
        providers: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Consolida as classificações de todos os provedores."""

        regressions = 0
        improvements = 0
        stable_metrics = 0
        not_comparable_metrics = 0

        providers_with_regressions = 0
        providers_with_improvements = 0
        providers_compared = 0

        for provider_result in providers.values():
            provider_status = provider_result.get("status")

            if provider_status in {
                "compared",
                "regression",
                "improvement",
                "mixed",
            }:
                providers_compared += 1

            provider_metrics = provider_result.get("metrics", {})

            metric_statuses: list[str] = []

            if isinstance(provider_metrics, Mapping):
                for metric in provider_metrics.values():
                    if not isinstance(metric, Mapping):
                        continue

                    status = metric.get("status")

                    metric_statuses.append(status)

                    if status == "regression":
                        regressions += 1

                    elif status == "improvement":
                        improvements += 1

                    elif status == "stable":
                        stable_metrics += 1

                    elif status == "not_comparable":
                        not_comparable_metrics += 1

            execution = provider_result.get("execution", {})

            execution_status = (
                execution.get("status") if isinstance(execution, Mapping) else None
            )

            execution_regression = execution_status == "regression"
            execution_improvement = execution_status == "improvement"

            # Se a execução falhou ou foi recuperada, suas métricas
            # não foram comparadas. Contabilizamos o evento de execução
            # uma única vez, sem duplicar o resultado por métrica.
            has_metric_regression = "regression" in metric_statuses
            has_metric_improvement = "improvement" in metric_statuses

            if (
                execution_regression
                and not has_metric_regression
                and not has_metric_improvement
            ):
                regressions += 1

            if (
                execution_improvement
                and not has_metric_regression
                and not has_metric_improvement
            ):
                improvements += 1

            provider_has_regression = has_metric_regression or execution_regression

            provider_has_improvement = has_metric_improvement or execution_improvement

            if provider_has_regression:
                providers_with_regressions += 1

            if provider_has_improvement:
                providers_with_improvements += 1

        if regressions and improvements:
            overall_status = "mixed"

        elif regressions:
            overall_status = "regression"

        elif improvements:
            overall_status = "improvement"

        elif providers_compared > 0 and stable_metrics > 0:
            overall_status = "stable"

        else:
            overall_status = "not_comparable"

        return {
            "status": overall_status,
            "providers_total": len(providers),
            "providers_compared": providers_compared,
            "providers_with_regressions": providers_with_regressions,
            "providers_with_improvements": providers_with_improvements,
            "regressions": regressions,
            "improvements": improvements,
            "stable_metrics": stable_metrics,
            "not_comparable_metrics": not_comparable_metrics,
        }

    # ======================================================
    # INFORMAÇÕES DOS BENCHMARKS
    # ======================================================

    @staticmethod
    def _benchmark_reference(
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Extrai os dados principais de uma execução."""

        dataset = report["dataset"]

        return {
            "benchmark_id": report["benchmark_id"],
            "created_at": report.get("created_at"),
            "dataset_sha256": dataset.get("sha256"),
            "dataset_case_count": dataset.get("case_count"),
        }

    # ======================================================
    # COMPARAÇÃO PRINCIPAL
    # ======================================================

    def compare(
        self,
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compara dois relatórios consolidados.

        Args:
            baseline: execução usada como referência.
            candidate: execução que será avaliada em relação à referência.

        Returns:
            Resultado estruturado com compatibilidade, resumo,
            métricas por provedor e diferenças encontradas.

        Raises:
            TypeError: quando um dos relatórios não é um mapeamento.
            ValueError: quando faltam dados obrigatórios ou
                quando as estruturas fundamentais são inválidas.
        """

        baseline = _require_report(
            baseline,
            "baseline",
        )

        candidate = _require_report(
            candidate,
            "candidate",
        )

        _validate_optional_report_sections(
            baseline,
            "baseline",
        )

        _validate_optional_report_sections(
            candidate,
            "candidate",
        )

        compatibility = _get_dataset_compatibility(
            baseline,
            candidate,
        )

        baseline_reference = self._benchmark_reference(baseline)

        candidate_reference = self._benchmark_reference(candidate)

        # --------------------------------------------------
        # DATASETS INCOMPATÍVEIS
        # --------------------------------------------------

        if not compatibility["comparable"]:
            return {
                "schema_version": 1,
                "baseline": baseline_reference,
                "candidate": candidate_reference,
                "compatibility": compatibility,
                "summary": {
                    "status": "incompatible",
                    "providers_total": 0,
                    "providers_compared": 0,
                    "providers_with_regressions": 0,
                    "providers_with_improvements": 0,
                    "regressions": 0,
                    "improvements": 0,
                    "stable_metrics": 0,
                    "not_comparable_metrics": 0,
                },
                "providers": {},
            }

        # --------------------------------------------------
        # PROVEDORES
        # --------------------------------------------------

        baseline_providers = _get_provider_names(baseline)

        candidate_providers = _get_provider_names(candidate)

        all_providers = sorted(baseline_providers | candidate_providers)

        provider_results: dict[str, dict[str, Any]] = {}

        for provider in all_providers:
            provider_results[provider] = self._compare_provider(
                provider=provider,
                baseline=baseline,
                candidate=candidate,
                baseline_exists=provider in baseline_providers,
                candidate_exists=provider in candidate_providers,
            )

        # --------------------------------------------------
        # RESUMO CONSOLIDADO
        # --------------------------------------------------

        summary = self._summarize_provider_results(provider_results)

        return {
            "schema_version": 1,
            "baseline": baseline_reference,
            "candidate": candidate_reference,
            "compatibility": compatibility,
            "summary": summary,
            "providers": provider_results,
        }
