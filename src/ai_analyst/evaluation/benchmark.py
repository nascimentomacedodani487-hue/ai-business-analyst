"""Benchmark comparativo de provedores do AI Business Analyst.

Executa o mesmo dataset em múltiplos provedores, preserva os
relatórios individuais e produz uma comparação consolidada.

O módulo não realiza chamadas diretamente às APIs. As chamadas
são delegadas ao EvaluationRunner. Isso permite injetar runners
simulados nos testes e evitar consumo de quotas.

A repetibilidade permite executar o mesmo dataset várias vezes
por provedor e calcular estatísticas de estabilidade.

A persistência histórica é opcional e delegada ao BenchmarkHistory,
injetado pelo construtor.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import fmean, median, stdev
from typing import Any
from uuid import uuid4

from ai_analyst.evaluation.dataset import (
    DATASET_VERSION,
    load_evaluation_dataset,
)
from ai_analyst.evaluation.history import BenchmarkHistory
from ai_analyst.evaluation.models import EvaluationCase
from ai_analyst.evaluation.runner import (
    SUPPORTED_PROVIDERS,
    EvaluationRunner,
)

logger = logging.getLogger(__name__)


BENCHMARK_SCHEMA_VERSION = 1

DEFAULT_BENCHMARK_PROVIDERS = tuple(sorted(SUPPORTED_PROVIDERS))


class BenchmarkRunner:
    """Orquestra benchmarks comparativos entre provedores de IA.

    Características:

    - Utiliza o mesmo conjunto de cenários para todos os provedores.
    - Cria uma instância independente de EvaluationRunner por execução.
    - Executa os provedores e repetições sequencialmente.
    - Isola falhas individuais para permitir a continuidade.
    - Permite configurar a quantidade de repetições.
    - Calcula estatísticas de repetibilidade por provedor.
    - Calcula estatísticas de estabilidade por cenário.
    - Gera fingerprint SHA-256 do dataset efetivamente avaliado.
    - Preserva os relatórios individuais das execuções.
    - Consolida métricas por provedor, categoria e cenário.
    - Separa custos completos de subtotais financeiros incompletos.
    - Permite persistir os resultados por injeção de dependência.

    O benchmark não aplica pesos arbitrários para determinar
    um vencedor. As métricas são apresentadas separadamente para
    permitir uma comparação transparente.

    Para preservar a compatibilidade, a execução padrão utiliza
    apenas uma repetição.
    """

    def __init__(
        self,
        providers: Iterable[str] = DEFAULT_BENCHMARK_PROVIDERS,
        dataset_path: str | Path | None = None,
        runner_factory: Callable[..., Any] | None = None,
        history_store: BenchmarkHistory | None = None,
        repetitions: int = 1,
    ):
        """Inicializa o benchmark."""

        if isinstance(providers, str):
            raise TypeError(
                "providers deve ser uma coleção de provedores, "
                "não uma string individual."
            )

        try:
            raw_providers = list(providers)
        except TypeError as exc:
            raise TypeError("providers deve ser uma coleção de strings.") from exc

        normalized_providers: list[str] = []

        for provider in raw_providers:
            if not isinstance(provider, str):
                raise TypeError("Todos os provedores devem ser strings.")

            normalized = provider.strip().lower()

            if not normalized:
                raise ValueError("O nome do provedor não pode ser vazio.")

            if normalized not in SUPPORTED_PROVIDERS:
                available = ", ".join(sorted(SUPPORTED_PROVIDERS))

                raise ValueError(
                    f"Provedor não suportado: '{provider}'. "
                    f"Provedores disponíveis: {available}."
                )

            if normalized in normalized_providers:
                raise ValueError(f"Provedor duplicado: '{normalized}'.")

            normalized_providers.append(normalized)

        if len(normalized_providers) < 2:
            raise ValueError(
                "O benchmark comparativo exige pelo menos dois provedores."
            )

        if isinstance(repetitions, bool) or not isinstance(repetitions, int):
            raise TypeError("repetitions deve ser um inteiro.")

        if repetitions < 1:
            raise ValueError("repetitions deve ser pelo menos 1.")

        self.providers = tuple(normalized_providers)

        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        self.runner_factory = (
            runner_factory if runner_factory is not None else EvaluationRunner
        )

        self.history_store = history_store
        self.repetitions = repetitions

    # ==========================================================
    # VALIDAÇÃO DOS CENÁRIOS
    # ==========================================================

    @staticmethod
    def _validate_cases(
        cases: Iterable[EvaluationCase],
    ) -> list[EvaluationCase]:
        """Valida os cenários que participarão do benchmark."""

        evaluation_cases = list(cases)

        if not evaluation_cases:
            raise ValueError("O benchmark exige pelo menos um cenário.")

        if not all(isinstance(case, EvaluationCase) for case in evaluation_cases):
            raise TypeError("Todos os cenários devem ser EvaluationCase.")

        case_ids = [case.case_id for case in evaluation_cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError("O dataset contém identificadores de cenários duplicados.")

        return evaluation_cases

    # ==========================================================
    # VALIDAÇÃO ESTATÍSTICA
    # ==========================================================

    @staticmethod
    def _is_valid_statistic(value: Any) -> bool:
        """Verifica se um valor pode participar dos cálculos estatísticos."""

        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
        )

    @classmethod
    def _summarize_numeric_samples(
        cls,
        values: Iterable[Any],
        requested_runs: int,
        *,
        decimal_places: int = 6,
    ) -> dict[str, Any]:
        """Calcula estatísticas descritivas de uma métrica.

        O desvio-padrão utilizado é o amostral, calculado somente
        quando existem pelo menos duas observações válidas.

        Valores ausentes, inválidos ou provenientes de execuções
        que não produziram a métrica não são incluídos no cálculo.

        A precisão padrão é de seis casas decimais. Métricas
        financeiras podem utilizar até doze casas decimais.
        """

        normalized_values = [
            float(value) for value in values if cls._is_valid_statistic(value)
        ]

        sample_count = len(normalized_values)

        if not normalized_values:
            return {
                "sample_count": 0,
                "missing_runs": requested_runs,
                "mean": None,
                "median": None,
                "standard_deviation": None,
                "min": None,
                "max": None,
            }

        average = fmean(normalized_values)
        median_value = median(normalized_values)

        standard_deviation = stdev(normalized_values) if sample_count >= 2 else None

        return {
            "sample_count": sample_count,
            "missing_runs": max(requested_runs - sample_count, 0),
            "mean": round(average, decimal_places),
            "median": round(median_value, decimal_places),
            "standard_deviation": (
                round(standard_deviation, decimal_places)
                if standard_deviation is not None
                else None
            ),
            "min": round(min(normalized_values), decimal_places),
            "max": round(max(normalized_values), decimal_places),
        }

    # ==========================================================
    # FINGERPRINT DO DATASET
    # ==========================================================

    @staticmethod
    def _calculate_dataset_fingerprint(
        cases: list[EvaluationCase],
    ) -> str:
        """Calcula SHA-256 do conteúdo efetivamente avaliado."""

        payload = {
            "version": DATASET_VERSION,
            "cases": [asdict(case) for case in cases],
        }

        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    # ==========================================================
    # SERIALIZAÇÃO E VALIDAÇÃO DOS RELATÓRIOS
    # ==========================================================

    @staticmethod
    def _serialize_runner_report(
        report: Any,
    ) -> dict[str, Any]:
        """Converte e valida o relatório do EvaluationRunner.

        Contrato mínimo:
        - run_id: identificador textual não vazio.
        - summary: objeto de mapeamento.
        - results: lista de resultados.
        - Cada resultado deve possuir case_id textual não vazio.

        metadata é opcional para preservar compatibilidade com
        relatórios legados.

        Um mapeamento vazio não é considerado um relatório válido.
        """

        if isinstance(report, Mapping):
            serialized = dict(report)

        else:
            to_dict = getattr(report, "to_dict", None)

            if not callable(to_dict):
                raise TypeError(
                    "O EvaluationRunner deve retornar um relatório "
                    "com método to_dict() ou um mapeamento."
                )

            serialized_report = to_dict()

            if not isinstance(serialized_report, Mapping):
                raise TypeError(
                    "O método to_dict() do relatório deve retornar "
                    "um objeto de mapeamento."
                )

            serialized = dict(serialized_report)

        run_id = serialized.get("run_id")

        if not isinstance(run_id, str) or not run_id.strip():
            raise TypeError(
                "O relatório do EvaluationRunner deve possuir "
                "run_id como string não vazia."
            )

        summary = serialized.get("summary")

        if not isinstance(summary, Mapping):
            raise TypeError(
                "O relatório do EvaluationRunner deve possuir "
                "summary como objeto de mapeamento."
            )

        results = serialized.get("results")

        if not isinstance(results, list):
            raise TypeError(
                "O relatório do EvaluationRunner deve possuir results como lista."
            )

        for index, result in enumerate(results):
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"O resultado {index} do relatório deve ser "
                    "um objeto de mapeamento."
                )

            case_id = result.get("case_id")

            if not isinstance(case_id, str) or not case_id.strip():
                raise TypeError(
                    f"O resultado {index} do relatório deve possuir "
                    "case_id como string não vazia."
                )

        metadata = serialized.get("metadata")

        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata do relatório deve ser um objeto de mapeamento.")

        return serialized

    # ==========================================================
    # EXECUÇÃO INDIVIDUAL DOS PROVEDORES
    # ==========================================================

    def _run_provider(
        self,
        provider: str,
        cases: list[EvaluationCase],
        benchmark_id: str,
        repetition: int | None = None,
    ) -> dict[str, Any]:
        """Executa o dataset em um provedor isoladamente.

        Uma nova instância de EvaluationRunner é criada em cada
        chamada deste método.

        Quando repetition não é informado, preserva o identificador
        original utilizado pelo benchmark anterior.
        """

        if repetition is None:
            run_id = f"{benchmark_id}-{provider}"
        else:
            run_id = f"{benchmark_id}-{provider}-rep-{repetition:03d}"

        logger.info(
            "Iniciando benchmark do provedor",
            extra={
                "event": "benchmark_provider_started",
                "benchmark_id": benchmark_id,
                "run_id": run_id,
                "provider": provider,
                "repetition": repetition,
                "scenario_count": len(cases),
            },
        )

        try:
            runner = self.runner_factory(
                provider_name=provider,
                dataset_path=self.dataset_path,
            )

            report = runner.run(
                cases=cases,
                run_id=run_id,
            )

            serialized_report = self._serialize_runner_report(report)

            logger.info(
                "Benchmark do provedor concluído",
                extra={
                    "event": "benchmark_provider_completed",
                    "benchmark_id": benchmark_id,
                    "run_id": run_id,
                    "provider": provider,
                    "repetition": repetition,
                },
            )

            return {
                "execution_status": "completed",
                "report": serialized_report,
                "error": None,
                "error_type": None,
            }

        except Exception as exc:
            logger.exception(
                "Falha no benchmark do provedor",
                extra={
                    "event": "benchmark_provider_failed",
                    "benchmark_id": benchmark_id,
                    "run_id": run_id,
                    "provider": provider,
                    "repetition": repetition,
                },
            )

            return {
                "execution_status": "error",
                "report": None,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    # ==========================================================
    # EXTRAÇÃO DE MÉTRICAS COMPARÁVEIS
    # ==========================================================

    @staticmethod
    def _metric_snapshot(
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Extrai métricas essenciais de um relatório individual.

        Suporta dois formatos:

        1. Relatório completo do EvaluationRunner:
           utiliza summary e metadata.metrics.summary.

        2. Resumo legado:
           utiliza diretamente metadata.metrics.summary.

        Quando os dois formatos estão presentes, os dados de
        summary do relatório têm prioridade.

        A taxa de aprovação é None quando não existem cenários
        efetivamente avaliados. Isso evita confundir ausência de
        avaliação com uma taxa real de aprovação de zero por cento.
        """

        run_summary = report.get("summary", {})

        metadata = report.get("metadata", {})

        metrics = metadata.get("metrics", {}) if isinstance(metadata, Mapping) else {}

        metric_summary = (
            metrics.get("summary", {}) if isinstance(metrics, Mapping) else {}
        )

        if not isinstance(run_summary, Mapping):
            run_summary = {}

        if not isinstance(metric_summary, Mapping):
            metric_summary = {}

        combined_summary = {
            **metric_summary,
            **run_summary,
        }

        total_cases = combined_summary.get("total_cases")

        evaluated_cases = combined_summary.get("evaluated_cases")

        error_cases = combined_summary.get("error_cases")

        skipped_cases = combined_summary.get("skipped_cases")

        valid_total = (
            isinstance(total_cases, int)
            and not isinstance(total_cases, bool)
            and total_cases > 0
        )

        valid_evaluated = (
            isinstance(evaluated_cases, int)
            and not isinstance(evaluated_cases, bool)
            and evaluated_cases >= 0
        )

        if valid_total:
            error_rate = (
                round(error_cases / total_cases * 100, 2)
                if (
                    isinstance(error_cases, int)
                    and not isinstance(error_cases, bool)
                    and 0 <= error_cases <= total_cases
                )
                else None
            )

            skipped_rate = (
                round(skipped_cases / total_cases * 100, 2)
                if (
                    isinstance(skipped_cases, int)
                    and not isinstance(skipped_cases, bool)
                    and 0 <= skipped_cases <= total_cases
                )
                else None
            )

        else:
            error_rate = None
            skipped_rate = None

        pass_rate = combined_summary.get("pass_rate")

        if valid_evaluated and evaluated_cases == 0:
            pass_rate = None

        elif (
            not valid_evaluated
            and isinstance(total_cases, int)
            and not isinstance(total_cases, bool)
            and total_cases == 0
        ):
            pass_rate = None

        token_cost_summary = report.get("token_cost_summary")

        if not isinstance(token_cost_summary, Mapping):
            token_cost_summary = None

        return {
            "total_cases": total_cases,
            "evaluated_cases": evaluated_cases,
            "passed_cases": combined_summary.get("passed_cases"),
            "failed_cases": combined_summary.get("failed_cases"),
            "error_cases": error_cases,
            "skipped_cases": skipped_cases,
            "pass_rate": pass_rate,
            "error_rate": error_rate,
            "skipped_rate": skipped_rate,
            "average_score": combined_summary.get("average_score"),
            "average_duration_ms": combined_summary.get("average_duration_ms"),
            "cache_hit_rate": combined_summary.get("cache_hit_rate"),
            "analysis_mode_adherence": combined_summary.get("analysis_mode_adherence"),
            "row_accuracy": combined_summary.get("row_accuracy"),
            "cell_accuracy": combined_summary.get("cell_accuracy"),
            "llm_usage": combined_summary.get("llm_usage"),
            "token_cost_summary": token_cost_summary,
        }

    # ==========================================================
    # EXTRAÇÃO DE ESTIMATIVAS FINANCEIRAS
    # ==========================================================

    @staticmethod
    def _extract_cost_summary(
        report: Mapping[str, Any],
        metric_snapshot: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Obtém a estimativa financeira da execução.

        Prioridade:
        1. report.token_cost_summary, produzido pelo EvaluationRunReport.
        2. metadata.metrics.summary.llm_usage.token_cost_estimate,
           para compatibilidade com relatórios sem resumo financeiro
           no nível principal.

        O resumo financeiro do EvaluationRunReport é a fonte
        preferencial porque considera todos os status de cenário.
        """

        report_costs = report.get("token_cost_summary")

        if isinstance(report_costs, Mapping):
            return report_costs

        llm_usage = metric_snapshot.get("llm_usage")

        if not isinstance(llm_usage, Mapping):
            return None

        usage_costs = llm_usage.get("token_cost_estimate")

        if isinstance(usage_costs, Mapping):
            return usage_costs

        return None

    # ==========================================================
    # REPETIBILIDADE POR CENÁRIO
    # ==========================================================

    @classmethod
    def _build_case_repeatability(
        cls,
        repetition_runs: list[dict[str, Any]],
        requested_runs: int,
    ) -> dict[str, dict[str, Any]]:
        """Calcula estabilidade das métricas de cada cenário.

        Compara o mesmo case_id entre as diferentes repetições.

        Métricas:
        - Taxa de aprovação.
        - Score.
        - Duração.
        - Taxa de cache.
        - Tokens de entrada, saída e totais.

        Cenários skipped não entram nas estatísticas de aprovação
        nem de score. Continuam contribuindo com duração, cache
        e consumo de tokens quando essas informações existem.
        """

        samples_by_case: dict[str, dict[str, Any]] = {}

        for repetition_run in repetition_runs:
            if repetition_run.get("execution_status") != "completed":
                continue

            report = repetition_run.get("report")

            if not isinstance(report, Mapping):
                continue

            results = report.get("results", [])

            if not isinstance(results, list):
                continue

            for result in results:
                if not isinstance(result, Mapping):
                    continue

                case_id = result.get("case_id")

                if not isinstance(case_id, str) or not case_id.strip():
                    continue

                if case_id not in samples_by_case:
                    samples_by_case[case_id] = {
                        "observed_runs": 0,
                        "passed_runs": 0,
                        "failed_or_error_runs": 0,
                        "skipped_runs": 0,
                        "pass_rate": [],
                        "score": [],
                        "duration_ms": [],
                        "cache_hit_rate": [],
                        "input_tokens": [],
                        "output_tokens": [],
                        "total_tokens": [],
                    }

                bucket = samples_by_case[case_id]
                bucket["observed_runs"] += 1

                status = result.get("status")

                if status == "passed":
                    bucket["passed_runs"] += 1
                    bucket["pass_rate"].append(100.0)

                elif status in ("failed", "error"):
                    bucket["failed_or_error_runs"] += 1
                    bucket["pass_rate"].append(0.0)

                elif status == "skipped":
                    bucket["skipped_runs"] += 1

                # Cenários skipped não possuem comparação válida.
                # Seu score padrão não deve reduzir artificialmente
                # a média de qualidade do cenário.
                if status in ("passed", "failed", "error"):
                    score = result.get("score")

                    if cls._is_valid_statistic(score):
                        bucket["score"].append(score)

                # Duração, cache e tokens continuam sendo registrados
                # mesmo quando o cenário termina como skipped.
                duration = result.get("duration_ms")

                if cls._is_valid_statistic(duration):
                    bucket["duration_ms"].append(duration)

                cache_hit = result.get("cache_hit")

                if isinstance(cache_hit, bool):
                    bucket["cache_hit_rate"].append(100.0 if cache_hit else 0.0)

                result_metadata = result.get("metadata", {})

                if not isinstance(result_metadata, Mapping):
                    continue

                llm_snapshot = result_metadata.get("llm_metrics")

                if not isinstance(llm_snapshot, Mapping):
                    continue

                tokens = llm_snapshot.get("tokens")

                if not isinstance(tokens, Mapping):
                    continue

                for token_field in (
                    "input",
                    "output",
                    "total",
                ):
                    token_value = tokens.get(token_field)

                    if cls._is_valid_statistic(token_value):
                        bucket[f"{token_field}_tokens"].append(token_value)

        comparison: dict[str, dict[str, Any]] = {}

        metric_names = (
            "pass_rate",
            "score",
            "duration_ms",
            "cache_hit_rate",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )

        for case_id, bucket in sorted(samples_by_case.items()):
            metrics = {
                metric_name: cls._summarize_numeric_samples(
                    bucket[metric_name],
                    requested_runs,
                )
                for metric_name in metric_names
            }

            observed_runs = bucket["observed_runs"]

            comparison[case_id] = {
                "requested_runs": requested_runs,
                "observed_runs": observed_runs,
                "missing_runs": max(requested_runs - observed_runs, 0),
                "passed_runs": bucket["passed_runs"],
                "failed_or_error_runs": bucket["failed_or_error_runs"],
                "skipped_runs": bucket["skipped_runs"],
                "metrics": metrics,
            }

        return comparison

    # ==========================================================
    # REPETIBILIDADE POR PROVEDOR
    # ==========================================================

    @classmethod
    def _build_repeatability_summary(
        cls,
        repetition_runs: list[dict[str, Any]],
        requested_runs: int,
    ) -> dict[str, Any]:
        """Consolida estatísticas das execuções repetidas de um provedor.

        Para cada execução bem-sucedida, extrai:
        - Taxa de aprovação.
        - Score médio.
        - Duração média.
        - Taxa de cache.
        - Custos financeiros completos.
        - Subtotais financeiros incompletos.
        - Tokens de entrada, saída e totais.

        Custos incompletos não participam da média de custo estimado.
        Seus subtotais conhecidos são preservados em uma métrica
        financeira separada.

        O desvio-padrão é amostral. Uma métrica com apenas
        uma observação válida terá desvio-padrão None.
        """

        completed_runs = [
            repetition
            for repetition in repetition_runs
            if (
                repetition.get("execution_status") == "completed"
                and isinstance(repetition.get("report"), Mapping)
            )
        ]

        failed_runs = requested_runs - len(completed_runs)

        metric_values: dict[str, list[float]] = {
            "pass_rate": [],
            "average_score": [],
            "average_duration_ms": [],
            "cache_hit_rate": [],
            "estimated_cost_usd": [],
            "known_cost_subtotal_usd": [],
            "input_tokens": [],
            "output_tokens": [],
            "total_tokens": [],
        }

        cost_observations = {
            "requested_runs": requested_runs,
            "observed_runs": 0,
            "complete_runs": 0,
            "incomplete_runs": 0,
            "missing_runs": 0,
        }

        for repetition in completed_runs:
            report = repetition["report"]

            snapshot = cls._metric_snapshot(report)

            for metric_name in (
                "pass_rate",
                "average_score",
                "average_duration_ms",
                "cache_hit_rate",
            ):
                value = snapshot.get(metric_name)

                if cls._is_valid_statistic(value):
                    metric_values[metric_name].append(float(value))

            llm_usage = snapshot.get("llm_usage")

            if isinstance(llm_usage, Mapping):
                tokens = llm_usage.get("tokens", {})

                if isinstance(tokens, Mapping):
                    for token_field in (
                        "input",
                        "output",
                        "total",
                    ):
                        value = tokens.get(token_field)

                        if cls._is_valid_statistic(value):
                            metric_values[f"{token_field}_tokens"].append(float(value))

            cost_summary = cls._extract_cost_summary(
                report,
                snapshot,
            )

            if not isinstance(cost_summary, Mapping):
                continue

            cost_observations["observed_runs"] += 1

            estimated_cost = cost_summary.get("estimated_cost_usd")

            complete_cost = cost_summary.get(
                "is_complete"
            ) is True and cls._is_valid_statistic(estimated_cost)

            if complete_cost:
                metric_values["estimated_cost_usd"].append(float(estimated_cost))

                cost_observations["complete_runs"] += 1

                continue

            # Um custo incompleto nunca entra na estatística
            # de estimated_cost_usd.
            #
            # Preservamos apenas o subtotal conhecido, quando
            # disponível, em uma série financeira separada.
            known_subtotal = cost_summary.get("known_subtotal_usd")

            if not cls._is_valid_statistic(known_subtotal):
                known_subtotal = estimated_cost

            if cls._is_valid_statistic(known_subtotal):
                metric_values["known_cost_subtotal_usd"].append(float(known_subtotal))

            cost_observations["incomplete_runs"] += 1

        cost_observations["missing_runs"] = max(
            requested_runs - cost_observations["observed_runs"],
            0,
        )

        metrics: dict[str, dict[str, Any]] = {}

        for metric_name, values in metric_values.items():
            decimal_places = (
                12
                if metric_name
                in (
                    "estimated_cost_usd",
                    "known_cost_subtotal_usd",
                )
                else 6
            )

            metrics[metric_name] = cls._summarize_numeric_samples(
                values,
                requested_runs,
                decimal_places=decimal_places,
            )

        case_repeatability = cls._build_case_repeatability(
            repetition_runs,
            requested_runs,
        )

        if failed_runs == 0:
            status = "completed"

        elif completed_runs:
            status = "completed_with_failures"

        else:
            status = "failed"

        representative_run_id = None

        if completed_runs:
            representative_report = completed_runs[0]["report"]
            representative_run_id = representative_report.get("run_id")

        return {
            "requested_runs": requested_runs,
            "completed_runs": len(completed_runs),
            "failed_runs": failed_runs,
            "status": status,
            "comparison_basis": "first_successful_repetition",
            "representative_run_id": representative_run_id,
            "standard_deviation_kind": "sample",
            "metrics": metrics,
            "cost_observations": cost_observations,
            "by_case": case_repeatability,
        }

    # ==========================================================
    # EXECUÇÃO COM REPETIBILIDADE
    # ==========================================================

    def _run_provider_repetitions(
        self,
        provider: str,
        cases: list[EvaluationCase],
        benchmark_id: str,
    ) -> dict[str, Any]:
        """Executa todas as repetições de um provedor.

        Cada repetição recebe:
        - Uma nova instância de EvaluationRunner.
        - Um run_id exclusivo.
        - O mesmo dataset validado.
        - O mesmo benchmark_id principal como referência.

        Uma falha em uma repetição não impede as seguintes.
        O primeiro relatório bem-sucedido é preservado como
        relatório representativo para manter a compatibilidade
        com as comparações tradicionais.
        """

        repetition_runs: list[dict[str, Any]] = []

        for repetition in range(1, self.repetitions + 1):
            logger.info(
                "Iniciando repetição do benchmark",
                extra={
                    "event": "benchmark_repetition_started",
                    "benchmark_id": benchmark_id,
                    "provider": provider,
                    "repetition": repetition,
                    "requested_runs": self.repetitions,
                },
            )

            execution = self._run_provider(
                provider=provider,
                cases=cases,
                benchmark_id=benchmark_id,
                repetition=repetition,
            )

            repetition_record = {
                "repetition": repetition,
                **execution,
            }

            repetition_runs.append(repetition_record)

        successful_runs = [
            repetition
            for repetition in repetition_runs
            if (
                repetition.get("execution_status") == "completed"
                and isinstance(repetition.get("report"), Mapping)
            )
        ]

        # Preserva o primeiro resultado bem-sucedido como referência
        # para os métodos tradicionais de comparação.
        representative = successful_runs[0] if successful_runs else repetition_runs[0]

        repeatability = self._build_repeatability_summary(
            repetition_runs=repetition_runs,
            requested_runs=self.repetitions,
        )

        return {
            "execution_status": ("completed" if successful_runs else "error"),
            "report": representative.get("report"),
            "error": (None if successful_runs else representative.get("error")),
            "error_type": (
                None if successful_runs else representative.get("error_type")
            ),
            "repetitions": repetition_runs,
            "repeatability": repeatability,
        }

    # ==========================================================
    # COMPARAÇÃO POR PROVEDOR
    # ==========================================================

    @classmethod
    def _build_provider_comparison(
        cls,
        provider_runs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Consolida as principais métricas por provedor."""

        comparison: dict[str, dict[str, Any]] = {}

        for provider, execution in provider_runs.items():
            if execution.get("execution_status") != "completed":
                comparison[provider] = {
                    "execution_status": "error",
                    "error": execution.get("error"),
                    "error_type": execution.get("error_type"),
                    "metrics": None,
                }

                if isinstance(execution.get("repeatability"), Mapping):
                    comparison[provider]["repeatability"] = execution["repeatability"]

                continue

            report = execution.get("report")

            if not isinstance(report, Mapping):
                comparison[provider] = {
                    "execution_status": "error",
                    "error": "Relatório do provedor inválido.",
                    "error_type": "InvalidBenchmarkReport",
                    "metrics": None,
                }

                if isinstance(execution.get("repeatability"), Mapping):
                    comparison[provider]["repeatability"] = execution["repeatability"]

                continue

            comparison[provider] = {
                "execution_status": "completed",
                "model": report.get("model"),
                "run_id": report.get("run_id"),
                "metrics": cls._metric_snapshot(report),
            }

            if isinstance(execution.get("repeatability"), Mapping):
                comparison[provider]["repeatability"] = execution["repeatability"]

        return comparison

    # ==========================================================
    # COMPARAÇÃO POR CATEGORIA
    # ==========================================================

    @classmethod
    def _build_category_comparison(
        cls,
        provider_runs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Compara métricas por categoria entre os provedores.

        Cada categoria utiliza o resumo produzido pelo próprio
        Evaluation Engine, preservando contagens, taxas, score,
        latência, cache, aderência ao planejamento, qualidade
        dos resultados e utilização de LLM.

        Em benchmarks repetidos, utiliza o relatório representativo
        selecionado pela análise de repetibilidade.
        """

        categories: dict[
            str,
            dict[str, dict[str, Any]],
        ] = {}

        for provider, execution in provider_runs.items():
            if execution.get("execution_status") != "completed":
                continue

            report = execution.get("report")

            if not isinstance(report, Mapping):
                continue

            metadata = report.get("metadata", {})

            if not isinstance(metadata, Mapping):
                continue

            metrics = metadata.get("metrics", {})

            if not isinstance(metrics, Mapping):
                continue

            by_category = metrics.get("by_category", {})

            if not isinstance(by_category, Mapping):
                continue

            for category, category_metrics in by_category.items():
                if not isinstance(category_metrics, Mapping):
                    continue

                synthetic_report = {
                    "metadata": {
                        "metrics": {
                            "summary": category_metrics,
                        },
                    },
                }

                categories.setdefault(
                    str(category),
                    {},
                )[provider] = cls._metric_snapshot(synthetic_report)

        return {
            category: dict(sorted(provider_metrics.items()))
            for category, provider_metrics in sorted(categories.items())
        }

    # ==========================================================
    # COMPARAÇÃO POR CENÁRIO
    # ==========================================================

    @staticmethod
    def _build_case_comparison(
        cases: list[EvaluationCase],
        provider_runs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Compara qualidade, execução e evidências por cenário.

        Preserva as métricas individuais produzidas pelo
        EvaluationRunner, sem recalcular valores.

        Também transporta os metadados de validação das
        evidências SQL para a comparação consolidada.

        Nos benchmarks repetidos, o relatório representativo
        é utilizado nesta comparação tradicional. As estatísticas
        das demais repetições ficam em repeatability.by_case.
        """

        comparison: dict[str, dict[str, Any]] = {}

        for case in cases:
            provider_results: dict[str, dict[str, Any]] = {}

            for provider, execution in provider_runs.items():
                report = execution.get("report")

                if execution.get("execution_status") != "completed" or not isinstance(
                    report, Mapping
                ):
                    provider_results[provider] = {
                        "status": "not_executed",
                        "score": None,
                        "duration_ms": None,
                        "analysis_mode": None,
                        "analysis_mode_matches": None,
                        "cache_hit": None,
                        "matched_rows": None,
                        "compared_rows": None,
                        "row_accuracy": None,
                        "matched_cells": None,
                        "compared_cells": None,
                        "cell_accuracy": None,
                        "llm_metrics": None,
                        "answer_evidence_validation": None,
                        "answer_evidence_status": None,
                        "answer_claims_supported": None,
                        "error": execution.get("error"),
                    }

                    continue

                results = report.get("results", [])

                if not isinstance(results, list):
                    results = []

                matching_result = next(
                    (
                        result
                        for result in results
                        if (
                            isinstance(result, Mapping)
                            and result.get("case_id") == case.case_id
                        )
                    ),
                    None,
                )

                if matching_result is None:
                    provider_results[provider] = {
                        "status": "missing_result",
                        "score": None,
                        "duration_ms": None,
                        "analysis_mode": None,
                        "analysis_mode_matches": None,
                        "cache_hit": None,
                        "matched_rows": None,
                        "compared_rows": None,
                        "row_accuracy": None,
                        "matched_cells": None,
                        "compared_cells": None,
                        "cell_accuracy": None,
                        "llm_metrics": None,
                        "answer_evidence_validation": None,
                        "answer_evidence_status": None,
                        "answer_claims_supported": None,
                        "error": (
                            "O relatório não contém resultado "
                            f"para o cenário '{case.case_id}'."
                        ),
                    }

                    continue

                result_metadata = matching_result.get("metadata", {})

                if not isinstance(result_metadata, Mapping):
                    result_metadata = {}

                provider_results[provider] = {
                    "status": matching_result.get("status"),
                    "score": matching_result.get("score"),
                    "duration_ms": matching_result.get("duration_ms"),
                    "analysis_mode": result_metadata.get("analysis_mode"),
                    "analysis_mode_matches": result_metadata.get(
                        "analysis_mode_matches"
                    ),
                    "cache_hit": matching_result.get("cache_hit"),
                    "matched_rows": result_metadata.get("matched_rows"),
                    "compared_rows": result_metadata.get("compared_rows"),
                    "row_accuracy": result_metadata.get("row_accuracy"),
                    "matched_cells": result_metadata.get("matched_cells"),
                    "compared_cells": result_metadata.get("compared_cells"),
                    "cell_accuracy": result_metadata.get("cell_accuracy"),
                    "llm_metrics": result_metadata.get("llm_metrics"),
                    "answer_evidence_validation": result_metadata.get(
                        "answer_evidence_validation"
                    ),
                    "answer_evidence_status": result_metadata.get(
                        "answer_evidence_status"
                    ),
                    "answer_claims_supported": result_metadata.get(
                        "answer_claims_supported"
                    ),
                    "error": matching_result.get("error"),
                }

            comparison[case.case_id] = {
                "category": case.category,
                "question": case.question,
                "expected_analysis_mode": case.expected_analysis_mode,
                "providers": provider_results,
            }

        return comparison

    # ==========================================================
    # PERSISTÊNCIA HISTÓRICA
    # ==========================================================

    def _persist_benchmark(
        self,
        benchmark_report: dict[str, Any],
    ) -> None:
        """Persiste o relatório quando existe um armazenamento injetado.

        Falhas de armazenamento são registradas e propagadas.
        O benchmark não é considerado persistido quando a operação
        do SQLite não termina com sucesso.
        """

        if self.history_store is None:
            return

        benchmark_id = benchmark_report["benchmark_id"]

        logger.info(
            "Persistindo benchmark no histórico",
            extra={
                "event": "benchmark_persistence_started",
                "benchmark_id": benchmark_id,
            },
        )

        try:
            self.history_store.save_benchmark(benchmark_report)

        except Exception:
            logger.exception(
                "Falha ao persistir benchmark no histórico",
                extra={
                    "event": "benchmark_persistence_failed",
                    "benchmark_id": benchmark_id,
                },
            )

            raise

        logger.info(
            "Benchmark persistido no histórico",
            extra={
                "event": "benchmark_persistence_completed",
                "benchmark_id": benchmark_id,
            },
        )

    # ==========================================================
    # EXECUÇÃO COMPLETA DO BENCHMARK
    # ==========================================================

    def run(
        self,
        cases: Iterable[EvaluationCase] | None = None,
        benchmark_id: str | None = None,
    ) -> dict[str, Any]:
        """Executa e consolida um benchmark comparativo.

        O dataset é carregado uma única vez e reutilizado
        por todos os provedores e repetições.

        Cada repetição recebe uma nova instância de EvaluationRunner.

        A execução é sequencial para evitar picos inesperados
        de consumo de API e facilitar a reprodução dos resultados.

        Uma falha individual não interrompe as demais repetições
        nem os outros provedores.

        Quando history_store é informado, o relatório final é
        persistido antes do retorno.
        """

        evaluation_cases = self._validate_cases(
            cases if cases is not None else load_evaluation_dataset(self.dataset_path)
        )

        normalized_benchmark_id = (
            benchmark_id if benchmark_id is not None else str(uuid4())
        )

        if (
            not isinstance(normalized_benchmark_id, str)
            or not normalized_benchmark_id.strip()
        ):
            raise ValueError("benchmark_id deve ser uma string não vazia.")

        dataset_fingerprint = self._calculate_dataset_fingerprint(evaluation_cases)

        created_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Benchmark comparativo iniciado",
            extra={
                "event": "benchmark_started",
                "benchmark_id": normalized_benchmark_id,
                "providers": self.providers,
                "scenario_count": len(evaluation_cases),
                "repetitions": self.repetitions,
                "dataset_sha256": dataset_fingerprint,
            },
        )

        provider_runs: dict[str, dict[str, Any]] = {}

        for provider in self.providers:
            if self.repetitions == 1:
                # Mantém o contrato antigo para o caso padrão,
                # sem adicionar estruturas de repetição.
                provider_runs[provider] = self._run_provider(
                    provider=provider,
                    cases=evaluation_cases,
                    benchmark_id=normalized_benchmark_id,
                )

            else:
                provider_runs[provider] = self._run_provider_repetitions(
                    provider=provider,
                    cases=evaluation_cases,
                    benchmark_id=normalized_benchmark_id,
                )

        provider_comparison = self._build_provider_comparison(provider_runs)

        category_comparison = self._build_category_comparison(provider_runs)

        case_comparison = self._build_case_comparison(
            evaluation_cases,
            provider_runs,
        )

        completed_providers = sum(
            execution["execution_status"] == "completed"
            for execution in provider_runs.values()
        )

        failed_providers = len(provider_runs) - completed_providers

        configuration = {
            "providers": list(self.providers),
            "provider_count": len(self.providers),
            "scenario_count": len(evaluation_cases),
            "execution_strategy": "sequential",
            "same_dataset_for_all_providers": True,
        }

        summary = {
            "provider_count": len(provider_runs),
            "completed_providers": completed_providers,
            "failed_providers": failed_providers,
            "all_providers_completed": failed_providers == 0,
        }

        benchmark_report = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark_id": normalized_benchmark_id,
            "created_at": created_at,
            "dataset": {
                "version": DATASET_VERSION,
                "source": (
                    str(self.dataset_path) if self.dataset_path is not None else None
                ),
                "case_count": len(evaluation_cases),
                "case_ids": [case.case_id for case in evaluation_cases],
                "sha256": dataset_fingerprint,
            },
            "configuration": configuration,
            "summary": summary,
            "provider_runs": provider_runs,
            "comparison": {
                "by_provider": provider_comparison,
                "by_category": category_comparison,
                "by_case": case_comparison,
            },
        }

        # ------------------------------------------------------
        # RELATÓRIO DE REPETIBILIDADE
        # ------------------------------------------------------

        if self.repetitions > 1:
            total_repetitions = len(self.providers) * self.repetitions

            completed_repetitions = sum(
                execution["repeatability"]["completed_runs"]
                for execution in provider_runs.values()
            )

            failed_repetitions = total_repetitions - completed_repetitions

            configuration["repetitions_per_provider"] = self.repetitions

            configuration["total_requested_runs"] = total_repetitions

            summary["total_repetitions"] = total_repetitions

            summary["completed_repetitions"] = completed_repetitions

            summary["failed_repetitions"] = failed_repetitions

            benchmark_report["repeatability"] = {
                "repetitions_per_provider": self.repetitions,
                "total_requested_runs": total_repetitions,
                "completed_runs": completed_repetitions,
                "failed_runs": failed_repetitions,
                "comparison_basis": "first_successful_repetition",
                "standard_deviation_kind": "sample",
                "by_provider": {
                    provider: execution["repeatability"]
                    for provider, execution in provider_runs.items()
                },
            }

        # ------------------------------------------------------
        # PERSISTÊNCIA
        # ------------------------------------------------------

        self._persist_benchmark(benchmark_report)

        logger.info(
            "Benchmark comparativo finalizado",
            extra={
                "event": "benchmark_completed",
                "benchmark_id": normalized_benchmark_id,
                "provider_count": len(provider_runs),
                "completed_providers": completed_providers,
                "failed_providers": failed_providers,
                "repetitions": self.repetitions,
                "persisted": self.history_store is not None,
            },
        )

        return benchmark_report
