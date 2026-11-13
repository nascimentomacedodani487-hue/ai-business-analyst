"""Benchmark comparativo de provedores do AI Business Analyst.

Executa o mesmo dataset em múltiplos provedores, preserva os
relatórios individuais e produz uma comparação consolidada.

O módulo não realiza chamadas diretamente às APIs. As chamadas
são delegadas ao EvaluationRunner. Isso permite injetar runners
simulados nos testes e evitar consumo de quotas.

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
from pathlib import Path
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
    - Cria uma instância independente de EvaluationRunner por provedor.
    - Executa os provedores sequencialmente.
    - Isola falhas individuais para permitir a continuidade.
    - Gera fingerprint SHA-256 do dataset efetivamente avaliado.
    - Preserva os relatórios individuais dos provedores.
    - Consolida métricas por provedor, categoria e cenário.
    - Permite persistir os resultados por injeção de dependência.

    O benchmark não aplica pesos arbitrários para determinar
    um vencedor. As métricas são apresentadas separadamente para
    permitir uma comparação transparente.
    """

    def __init__(
        self,
        providers: Iterable[str] = DEFAULT_BENCHMARK_PROVIDERS,
        dataset_path: str | Path | None = None,
        runner_factory: Callable[..., Any] | None = None,
        history_store: BenchmarkHistory | None = None,
    ):
        """Inicializa o benchmark.

        Args:
            providers: Provedores que participarão da comparação.
                São necessários pelo menos dois provedores.
            dataset_path: Caminho opcional do dataset.
            runner_factory: Fábrica injetável de EvaluationRunner.
                Deve aceitar provider_name e dataset_path.
            history_store: Armazenamento histórico opcional.
                Quando informado, o relatório consolidado é persistido
                antes de ser retornado ao chamador.

        Raises:
            TypeError: Quando os provedores não são uma coleção
                válida de strings.
            ValueError: Quando existem provedores inválidos,
                duplicados ou menos de dois provedores.
        """

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

        self.providers = tuple(normalized_providers)

        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        self.runner_factory = (
            runner_factory if runner_factory is not None else EvaluationRunner
        )

        self.history_store = history_store

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
    # FINGERPRINT DO DATASET
    # ==========================================================

    @staticmethod
    def _calculate_dataset_fingerprint(
        cases: list[EvaluationCase],
    ) -> str:
        """Calcula SHA-256 do conteúdo efetivamente avaliado.

        A serialização canônica considera todos os campos dos
        cenários, incluindo perguntas, SQLs, tolerâncias, colunas
        obrigatórias, tags e modo de análise esperado.

        Alterações na formatação do JSON original não alteram
        o fingerprint quando o conteúdo semântico permanece igual.
        """

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
    # SERIALIZAÇÃO DOS RELATÓRIOS
    # ==========================================================

    @staticmethod
    def _serialize_runner_report(
        report: Any,
    ) -> dict[str, Any]:
        """Converte o relatório do EvaluationRunner em dicionário."""

        if isinstance(report, Mapping):
            return dict(report)

        to_dict = getattr(
            report,
            "to_dict",
            None,
        )

        if not callable(to_dict):
            raise TypeError(
                "O EvaluationRunner deve retornar um relatório "
                "com método to_dict() ou um mapeamento."
            )

        serialized = to_dict()

        if not isinstance(serialized, Mapping):
            raise TypeError(
                "O método to_dict() do relatório deve retornar um objeto de mapeamento."
            )

        return dict(serialized)

    # ==========================================================
    # EXECUÇÃO INDIVIDUAL DOS PROVEDORES
    # ==========================================================

    def _run_provider(
        self,
        provider: str,
        cases: list[EvaluationCase],
        benchmark_id: str,
    ) -> dict[str, Any]:
        """Executa o dataset em um provedor isoladamente."""

        run_id = f"{benchmark_id}-{provider}"

        logger.info(
            "Iniciando benchmark do provedor",
            extra={
                "event": "benchmark_provider_started",
                "benchmark_id": benchmark_id,
                "run_id": run_id,
                "provider": provider,
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

        2. Resumo segmentado por categoria:
           utiliza diretamente os dados recebidos em
           metadata.metrics.summary.

        Quando os dois formatos estão presentes, os dados de
        summary do relatório têm prioridade para contagens e
        taxa de aprovação.

        Métricas de qualidade e utilização de LLM são preservadas
        diretamente do resumo consolidado:
        - row_accuracy;
        - cell_accuracy;
        - llm_usage.

        Os novos campos são opcionais para manter compatibilidade
        com relatórios de versões anteriores do Evaluation Engine.
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

        error_cases = combined_summary.get("error_cases")

        skipped_cases = combined_summary.get("skipped_cases")

        if (
            isinstance(total_cases, int)
            and not isinstance(total_cases, bool)
            and total_cases > 0
        ):
            error_rate = (
                round(error_cases / total_cases * 100, 2)
                if isinstance(error_cases, int)
                and not isinstance(error_cases, bool)
                and error_cases >= 0
                else None
            )

            skipped_rate = (
                round(skipped_cases / total_cases * 100, 2)
                if isinstance(skipped_cases, int)
                and not isinstance(skipped_cases, bool)
                and skipped_cases >= 0
                else None
            )

        else:
            error_rate = None
            skipped_rate = None

        return {
            "total_cases": total_cases,
            "evaluated_cases": combined_summary.get("evaluated_cases"),
            "passed_cases": combined_summary.get("passed_cases"),
            "failed_cases": combined_summary.get("failed_cases"),
            "error_cases": error_cases,
            "skipped_cases": skipped_cases,
            "pass_rate": combined_summary.get("pass_rate"),
            "error_rate": error_rate,
            "skipped_rate": skipped_rate,
            "average_score": combined_summary.get("average_score"),
            "average_duration_ms": combined_summary.get("average_duration_ms"),
            "cache_hit_rate": combined_summary.get("cache_hit_rate"),
            "analysis_mode_adherence": combined_summary.get("analysis_mode_adherence"),
            "row_accuracy": combined_summary.get("row_accuracy"),
            "cell_accuracy": combined_summary.get("cell_accuracy"),
            "llm_usage": combined_summary.get("llm_usage"),
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

                continue

            report = execution.get("report")

            if not isinstance(report, Mapping):
                comparison[provider] = {
                    "execution_status": "error",
                    "error": "Relatório do provedor inválido.",
                    "error_type": "InvalidBenchmarkReport",
                    "metrics": None,
                }

                continue

            comparison[provider] = {
                "execution_status": "completed",
                "model": report.get("model"),
                "run_id": report.get("run_id"),
                "metrics": cls._metric_snapshot(report),
            }

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
        """Compara qualidade, execução e consumo de LLM por cenário.

        Preserva as métricas individuais produzidas pelo
        EvaluationRunner, sem recalcular valores.

        Cada provedor apresenta:
        - Status, score e duração.
        - Modo de análise e aderência ao modo esperado.
        - Indicadores de cache e erro.
        - Linhas e células comparadas e correspondentes.
        - Precisão por linha e por célula.
        - Métricas individuais de utilização de LLM.

        Campos indisponíveis são representados por None,
        mantendo um contrato consistente entre provedores.
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
                        if isinstance(result, Mapping)
                        and result.get("case_id") == case.case_id
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
                        "error": (
                            "O relatório não contém resultado "
                            f"para o cenário '{case.case_id}'."
                        ),
                    }

                    continue

                result_metadata = matching_result.get(
                    "metadata",
                    {},
                )

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
        por todos os provedores.

        Cada provedor recebe um EvaluationRunner independente.

        A execução é sequencial para evitar picos inesperados
        de consumo de API e facilitar a reprodução dos resultados.

        Uma falha individual não interrompe os demais provedores.

        Quando history_store é informado, o relatório final é
        persistido antes do retorno.

        Args:
            cases: Cenários opcionais. Quando omitido, utiliza
                o dataset padrão ou o caminho configurado.
            benchmark_id: Identificador opcional do benchmark.

        Returns:
            Dicionário contendo:
            - Identificação e fingerprint do dataset.
            - Configuração do benchmark.
            - Relatórios individuais dos provedores.
            - Comparação por provedor, categoria e cenário.

        Raises:
            ValueError: Quando o benchmark_id já existe no histórico
                ou quando o relatório não atende ao contrato de persistência.
            sqlite3.Error: Quando ocorre falha de armazenamento SQLite.
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
                "dataset_sha256": dataset_fingerprint,
            },
        )

        provider_runs: dict[str, dict[str, Any]] = {}

        for provider in self.providers:
            provider_runs[provider] = self._run_provider(
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
            "configuration": {
                "providers": list(self.providers),
                "provider_count": len(self.providers),
                "scenario_count": len(evaluation_cases),
                "execution_strategy": "sequential",
                "same_dataset_for_all_providers": True,
            },
            "summary": {
                "provider_count": len(provider_runs),
                "completed_providers": completed_providers,
                "failed_providers": failed_providers,
                "all_providers_completed": failed_providers == 0,
            },
            "provider_runs": provider_runs,
            "comparison": {
                "by_provider": provider_comparison,
                "by_category": category_comparison,
                "by_case": case_comparison,
            },
        }

        # A persistência é opcional e ocorre somente quando
        # o chamador injeta explicitamente um BenchmarkHistory.
        #
        # O relatório é armazenado antes do retorno.
        # Qualquer falha de persistência é propagada para o chamador.
        self._persist_benchmark(benchmark_report)

        logger.info(
            "Benchmark comparativo finalizado",
            extra={
                "event": "benchmark_completed",
                "benchmark_id": normalized_benchmark_id,
                "provider_count": len(provider_runs),
                "completed_providers": completed_providers,
                "failed_providers": failed_providers,
                "persisted": self.history_store is not None,
            },
        )

        return benchmark_report
