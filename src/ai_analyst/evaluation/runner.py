"""Orquestração das avaliações do AI Business Analyst."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from ai_analyst.agent.business_analyst import BusinessAnalyst
from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.evaluation.dataset import load_evaluation_dataset
from ai_analyst.evaluation.evaluator import ResultEvaluator
from ai_analyst.evaluation.metrics import calculate_evaluation_metrics
from ai_analyst.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationRunReport,
)
from ai_analyst.llm.metrics import llm_metrics

logger = logging.getLogger(__name__)


SUPPORTED_PROVIDERS = {
    "gemini",
    "openai",
    "anthropic",
}


SUPPORTED_ANALYSIS_MODES = {
    "single_query",
    "multi_query",
}


class EvaluationRunner:
    """Executa cenários e consolida os resultados das avaliações.

    O runner mantém o BusinessAnalyst independente.

    As consultas de referência são executadas pelo SQLExecutor,
    que utiliza os guardrails e a conexão somente leitura existentes.

    As perguntas avaliadas são executadas pelo BusinessAnalyst,
    utilizando o provedor selecionado.

    O runner aceita dependências injetáveis para permitir testes
    sem chamadas reais aos provedores de IA.

    Cada cenário possui um contexto independente de métricas LLM,
    permitindo rastrear requisições, tokens, retries e latência.
    """

    def __init__(
        self,
        provider_name: str = "gemini",
        dataset_path: str | Path | None = None,
        analyst_factory: Callable[..., Any] | None = None,
        reference_executor: SQLExecutor | None = None,
        evaluator: ResultEvaluator | None = None,
    ):
        normalized_provider = provider_name.strip().lower()

        if normalized_provider not in SUPPORTED_PROVIDERS:
            available = ", ".join(sorted(SUPPORTED_PROVIDERS))

            raise ValueError(
                f"Provedor não suportado: '{provider_name}'. "
                f"Provedores disponíveis: {available}."
            )

        self.provider_name = normalized_provider

        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        self.analyst_factory = analyst_factory or BusinessAnalyst

        self.reference_executor = (
            reference_executor if reference_executor is not None else SQLExecutor()
        )

        self.evaluator = evaluator if evaluator is not None else ResultEvaluator()

    # ==========================================================
    # INICIALIZAÇÃO
    # ==========================================================

    def _create_analyst(self):
        """Cria uma nova instância do agente."""

        return self.analyst_factory(provider_name=self.provider_name)

    # ==========================================================
    # MÉTRICAS INDIVIDUAIS DE LLM
    # ==========================================================

    @staticmethod
    def _snapshot_llm_metrics(
        analysis_metrics: Any,
        *,
        provider: str,
        model: str | None = None,
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        """Obtém um snapshot das métricas de LLM do cenário.

        Utiliza o contexto individual criado por LLMMetrics.track_analysis().

        O provedor vem da configuração efetiva do runner, evitando
        que o valor padrão do acumulador seja confundido com o
        provedor realmente selecionado.

        Quando não existem chamadas registradas:
        - O modelo é obtido do BusinessAnalyst.
        - Se o modelo não estiver disponível, utiliza 'unknown'.

        Quando existe um analysis_id retornado pelo agente,
        ele substitui o identificador temporário de rastreamento.
        """

        snapshot = analysis_metrics.snapshot()

        if not isinstance(snapshot, Mapping):
            raise TypeError(
                "O snapshot de métricas LLM deve ser um objeto de mapeamento."
            )

        normalized_snapshot = dict(snapshot)

        requests = normalized_snapshot.get("requests", {})

        if not isinstance(requests, Mapping):
            requests = {}

        requests_total = requests.get("total", 0)

        has_recorded_requests = (
            isinstance(requests_total, int)
            and not isinstance(requests_total, bool)
            and requests_total > 0
        )

        normalized_snapshot["provider"] = provider

        if not has_recorded_requests:
            normalized_snapshot["model"] = model or "unknown"

        elif not normalized_snapshot.get("model"):
            normalized_snapshot["model"] = model or "unknown"

        if analysis_id:
            normalized_snapshot["analysis_id"] = analysis_id

        return normalized_snapshot

    # ==========================================================
    # NORMALIZAÇÃO DOS RESULTADOS
    # ==========================================================

    @staticmethod
    def _validate_analysis_rows(
        rows: Any,
        source: str,
    ) -> list[dict[str, Any]]:
        """Valida e normaliza uma lista de resultados."""

        if not isinstance(rows, list):
            raise TypeError(f"Os resultados de '{source}' devem ser uma lista.")

        normalized_rows = []

        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"O registro {index} de '{source}' "
                    "deve ser um objeto de mapeamento."
                )

            if not all(isinstance(column, str) for column in row):
                raise TypeError(
                    f"Todas as colunas de '{source}' devem possuir nomes string."
                )

            normalized_rows.append(dict(row))

        return normalized_rows

    @staticmethod
    def _get_columns(
        rows: list[dict[str, Any]],
    ) -> set[str]:
        """Obtém as colunas disponíveis em uma tabela de resultados."""

        columns: set[str] = set()

        for row in rows:
            columns.update(row.keys())

        return columns

    # ==========================================================
    # COMBINAÇÃO SEGURA DE RESULTADOS ESCALARES
    # ==========================================================

    @staticmethod
    def _merge_scalar_results(
        tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Combina resultados escalares sem estabelecer joins.

        Cada consulta precisa retornar exatamente uma linha.

        Colunas repetidas são aceitas somente quando possuem
        valores iguais. Resultados conflitantes não são combinados.

        Não tenta estabelecer relacionamentos entre tabelas.
        """

        merged: dict[str, Any] = {}

        for table in tables:
            rows = table["rows"]

            if len(rows) != 1:
                return None

            row = rows[0]

            for column, value in row.items():
                if column in merged and merged[column] != value:
                    return None

                merged[column] = value

        return [merged]

    # ==========================================================
    # EXTRAÇÃO DE ANÁLISES COMPOSTAS
    # ==========================================================

    @classmethod
    def _extract_multi_query_rows(
        cls,
        queries: list[dict[str, Any]],
        expected_rows: list[dict[str, Any]],
        case: EvaluationCase,
    ) -> tuple[
        list[dict[str, Any]] | None,
        dict[str, Any],
    ]:
        """Obtém resultados comparáveis de uma análise composta.

        Estratégias permitidas:

        1. Combinar resultados escalares de uma linha, desde que
           não existam valores conflitantes em colunas repetidas.
        2. Utilizar uma consulta que já contenha todas as colunas
           necessárias para a avaliação.

        Não realiza joins heurísticos entre tabelas.
        """

        successful_queries = [
            query for query in queries if query.get("status") == "success"
        ]

        failed_queries = [
            query
            for query in queries
            if query.get("status") == "failure"
            and query.get("failure_type") != "unsupported_capability"
        ]

        unsupported_queries = [
            query
            for query in queries
            if query.get("failure_type") == "unsupported_capability"
        ]

        metadata = {
            "query_count": len(queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "unsupported_queries": len(unsupported_queries),
            "partial_success": (
                len(successful_queries) > 0
                and (len(failed_queries) > 0 or len(unsupported_queries) > 0)
            ),
        }

        if not successful_queries:
            if queries and len(unsupported_queries) == len(queries):
                metadata["reason"] = "unsupported_capability"

            else:
                metadata["reason"] = "no_successful_queries"

            return None, metadata

        tables = []

        for query in successful_queries:
            rows = cls._validate_analysis_rows(
                query.get("results", []),
                f"query_{query.get('id')}",
            )

            tables.append(
                {
                    "query_id": query.get("id"),
                    "rows": rows,
                    "columns": cls._get_columns(rows),
                }
            )

        target_columns = set(case.required_columns) or cls._get_columns(expected_rows)

        # ------------------------------------------------------
        # PRIMEIRO: verificar resultados escalares.
        #
        # Esta verificação precisa acontecer antes de selecionar
        # uma consulta completa. Caso contrário, uma consulta
        # poderia ocultar valores conflitantes de outra consulta.
        # ------------------------------------------------------

        all_scalar = all(len(table["rows"]) == 1 for table in tables)

        if all_scalar:
            merged = cls._merge_scalar_results(tables)

            if merged is None:
                metadata["reason"] = "multi_query_results_not_combinable"

                metadata["conflict_type"] = "conflicting_scalar_values"

                metadata["selected_query_ids"] = [table["query_id"] for table in tables]

                return None, metadata

            merged_columns = cls._get_columns(merged)

            metadata["selected_query_ids"] = [table["query_id"] for table in tables]

            metadata["results_combined"] = len(tables) > 1

            if target_columns:
                metadata["missing_columns"] = sorted(target_columns - merged_columns)

            return merged, metadata

        # ------------------------------------------------------
        # SEGUNDO: sem resultados exclusivamente escalares,
        # procurar uma consulta que já contenha todas as
        # colunas necessárias.
        # ------------------------------------------------------

        if not target_columns:
            if len(tables) == 1:
                selected = tables[0]

                metadata["selected_query_ids"] = [selected["query_id"]]

                return selected["rows"], metadata

            metadata["reason"] = "insufficient_reference_columns"

            return None, metadata

        complete_tables = [
            table for table in tables if target_columns.issubset(table["columns"])
        ]

        if complete_tables:
            selected = min(
                complete_tables,
                key=lambda table: (
                    len(table["columns"] - target_columns),
                    -len(table["rows"]),
                ),
            )

            metadata["selected_query_ids"] = [selected["query_id"]]

            return selected["rows"], metadata

        # ------------------------------------------------------
        # TERCEIRO: uma única consulta incompleta pode ser
        # encaminhada ao ResultEvaluator. Ele identificará
        # as colunas ausentes e reprovará a comparação.
        # ------------------------------------------------------

        if len(tables) == 1:
            selected = tables[0]

            metadata["selected_query_ids"] = [selected["query_id"]]

            metadata["missing_columns"] = sorted(target_columns - selected["columns"])

            return selected["rows"], metadata

        # ------------------------------------------------------
        # QUARTO: não combinar tabelas multidimensionais
        # por chaves ou colunas escolhidas heuristicamente.
        # ------------------------------------------------------

        metadata["reason"] = "multi_query_results_not_combinable"

        metadata["required_columns"] = sorted(target_columns)

        return None, metadata

    # ==========================================================
    # VALIDAÇÃO DO MODO DE ANÁLISE
    # ==========================================================

    @staticmethod
    def _apply_analysis_mode_expectation(
        result: EvaluationCaseResult,
        case: EvaluationCase,
        actual_mode: str,
    ) -> EvaluationCaseResult:
        """Compara o modo executado com o modo esperado.

        A validação do modo de análise é independente da
        comparação dos resultados SQL.

        Quando existe divergência:
        - O cenário é reprovado.
        - A pontuação do ResultEvaluator é preservada.
        - O status original da comparação é registrado.
        - A divergência fica disponível nos metadados.

        Falhas operacionais continuam com status 'error'.
        """

        expected_mode = case.expected_analysis_mode

        # Cenários sem expectativa explícita preservam
        # integralmente o comportamento anterior do runner.
        if expected_mode is None:
            return result

        mode_matches = actual_mode == expected_mode

        result.metadata.update(
            {
                "analysis_mode": actual_mode,
                "expected_analysis_mode": expected_mode,
                "analysis_mode_matches": mode_matches,
            }
        )

        if mode_matches:
            return result

        result.metadata["analysis_mode_mismatch"] = {
            "expected": expected_mode,
            "actual": actual_mode,
        }

        # Uma divergência de modo é uma reprovação de qualidade,
        # não uma falha operacional.
        #
        # Se a comparação SQL ou extração já gerou um erro
        # operacional, o status 'error' deve ser preservado.
        if result.status != "error":
            result.metadata["result_comparison_status"] = result.status

            result.metadata["result_comparison_score"] = result.score

            result.status = "failed"

        return result

    # ==========================================================
    # ERROS OPERACIONAIS
    # ==========================================================

    @staticmethod
    def _error_result(
        case: EvaluationCase,
        error: Exception,
        *,
        phase: str,
        provider: str,
        model: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationCaseResult:
        """Monta um resultado de avaliação com falha operacional."""

        result_metadata = {
            "category": case.category,
            "phase": phase,
            "error_type": type(error).__name__,
        }

        if case.expected_analysis_mode is not None:
            result_metadata["expected_analysis_mode"] = case.expected_analysis_mode

        if metadata:
            result_metadata.update(metadata)

        return EvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            status="error",
            score=0.0,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            error=str(error),
            metadata=result_metadata,
        )

    # ==========================================================
    # FINALIZAÇÃO E MÉTRICAS
    # ==========================================================

    @staticmethod
    def _finalize_report(
        report: EvaluationRunReport,
    ) -> EvaluationRunReport:
        """Calcula métricas e finaliza o relatório."""

        report.metadata["metrics"] = calculate_evaluation_metrics(report.results)

        report.finalize()

        return report

    # ==========================================================
    # EXECUÇÃO DE UM CENÁRIO
    # ==========================================================

    def _run_case(
        self,
        case: EvaluationCase,
        analyst: Any,
    ) -> EvaluationCaseResult:
        """Executa um cenário individual."""

        # ------------------------------------------------------
        # 1. Calcular o resultado de referência.
        #
        # A consulta de referência não pertence ao contexto
        # de métricas LLM e não deve contabilizar consumo de IA.
        # ------------------------------------------------------

        reference_start = perf_counter()

        try:
            expected_rows = self.reference_executor.execute(case.reference_sql)

            expected_rows = self._validate_analysis_rows(
                expected_rows,
                "reference_sql",
            )

            reference_sql_ms = round(
                (perf_counter() - reference_start) * 1000,
                2,
            )

        except Exception as exc:
            reference_sql_ms = round(
                (perf_counter() - reference_start) * 1000,
                2,
            )

            logger.exception(
                "Falha ao executar SQL de referência",
                extra={
                    "event": "evaluation_reference_sql_failed",
                    "case_id": case.case_id,
                },
            )

            return self._error_result(
                case,
                exc,
                phase="reference_sql",
                provider=self.provider_name,
                model=getattr(analyst, "model", None),
                duration_ms=reference_sql_ms,
            )

        # ------------------------------------------------------
        # 2. Executar a pergunta no agente.
        #
        # Cada cenário recebe um contexto independente.
        # Isso impede que tokens e latências de cenários
        # anteriores sejam incorporados ao cenário atual.
        # ------------------------------------------------------

        analysis_tracking_id = str(uuid4())

        analysis_metrics = None

        analysis_start = perf_counter()

        try:
            with llm_metrics.track_analysis(
                analysis_id=analysis_tracking_id,
            ) as analysis_metrics:
                analysis = analyst.analyze(case.question)

        except Exception as exc:
            analysis_duration_ms = round(
                (perf_counter() - analysis_start) * 1000,
                2,
            )

            llm_usage = None

            if analysis_metrics is not None:
                llm_usage = self._snapshot_llm_metrics(
                    analysis_metrics,
                    provider=self.provider_name,
                    model=getattr(analyst, "model", None),
                    analysis_id=analysis_tracking_id,
                )

            logger.exception(
                "Falha durante execução do agente",
                extra={
                    "event": "evaluation_agent_execution_failed",
                    "case_id": case.case_id,
                    "provider": self.provider_name,
                },
            )

            error_metadata = {
                "reference_sql_ms": reference_sql_ms,
            }

            if llm_usage is not None:
                error_metadata["llm_metrics"] = llm_usage

            return self._error_result(
                case,
                exc,
                phase="agent_execution",
                provider=self.provider_name,
                model=getattr(analyst, "model", None),
                duration_ms=analysis_duration_ms,
                metadata=error_metadata,
            )

        analysis_duration_ms = round(
            (perf_counter() - analysis_start) * 1000,
            2,
        )

        # ------------------------------------------------------
        # 3. Validar o retorno do agente.
        # ------------------------------------------------------

        if not isinstance(analysis, Mapping):
            error = TypeError("O BusinessAnalyst deve retornar um dicionário.")

            llm_usage = self._snapshot_llm_metrics(
                analysis_metrics,
                provider=self.provider_name,
                model=getattr(analyst, "model", None),
                analysis_id=analysis_tracking_id,
            )

            return self._error_result(
                case,
                error,
                phase="analysis_validation",
                provider=self.provider_name,
                model=getattr(analyst, "model", None),
                duration_ms=analysis_duration_ms,
                metadata={
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

        actual_model = analysis.get("model") or getattr(analyst, "model", None)

        actual_provider = analysis.get("provider") or self.provider_name

        actual_analysis_id = analysis.get("analysis_id") or analysis_tracking_id

        llm_usage = self._snapshot_llm_metrics(
            analysis_metrics,
            provider=actual_provider,
            model=actual_model,
            analysis_id=actual_analysis_id,
        )

        analysis_mode = analysis.get(
            "analysis_mode",
            "single_query",
        )

        extraction_metadata: dict[str, Any] = {}

        # ------------------------------------------------------
        # 4. Extrair resultados simples ou compostos.
        # ------------------------------------------------------

        try:
            if analysis_mode == "multi_query":
                queries = analysis.get("queries", [])

                if not isinstance(queries, list):
                    raise TypeError(
                        "O campo 'queries' de uma análise composta deve ser uma lista."
                    )

                actual_rows, extraction_metadata = self._extract_multi_query_rows(
                    queries=queries,
                    expected_rows=expected_rows,
                    case=case,
                )

                if actual_rows is None:
                    reason = extraction_metadata.get(
                        "reason",
                        "multi_query_results_not_available",
                    )

                    skipped_reasons = {
                        "unsupported_capability",
                        "multi_query_results_not_combinable",
                        "required_columns_not_available",
                        "insufficient_reference_columns",
                    }

                    status = "skipped" if reason in skipped_reasons else "error"

                    result = EvaluationCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        status=status,
                        score=0.0,
                        expected_rows=expected_rows,
                        actual_rows=[],
                        provider=actual_provider,
                        model=actual_model,
                        duration_ms=analysis_duration_ms,
                        cache_hit=analysis.get("cache_hit"),
                        error=reason,
                        metadata={
                            "category": case.category,
                            "analysis_mode": analysis_mode,
                            "reference_sql_ms": reference_sql_ms,
                            "llm_metrics": llm_usage,
                            **extraction_metadata,
                        },
                    )

                    return self._apply_analysis_mode_expectation(
                        result,
                        case,
                        analysis_mode,
                    )

            else:
                actual_rows = self._validate_analysis_rows(
                    analysis.get("results", []),
                    "agent_results",
                )

        except Exception as exc:
            logger.exception(
                "Falha ao extrair resultados do agente",
                extra={
                    "event": "evaluation_result_extraction_failed",
                    "case_id": case.case_id,
                },
            )

            result = self._error_result(
                case,
                exc,
                phase="result_extraction",
                provider=actual_provider,
                model=actual_model,
                duration_ms=analysis_duration_ms,
                metadata={
                    "analysis_mode": analysis_mode,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

            return self._apply_analysis_mode_expectation(
                result,
                case,
                analysis_mode,
            )

        # ------------------------------------------------------
        # 5. Comparar resultado real com referência.
        # ------------------------------------------------------

        try:
            evaluation_result = self.evaluator.evaluate(
                case=case,
                expected_rows=expected_rows,
                actual_rows=actual_rows,
                analysis_id=analysis.get("analysis_id"),
                provider=actual_provider,
                model=actual_model,
                duration_ms=analysis_duration_ms,
                cache_hit=analysis.get("cache_hit"),
            )

        except Exception as exc:
            logger.exception(
                "Falha durante a comparação de resultados",
                extra={
                    "event": "evaluation_comparison_failed",
                    "case_id": case.case_id,
                },
            )

            result = self._error_result(
                case,
                exc,
                phase="result_comparison",
                provider=actual_provider,
                model=actual_model,
                duration_ms=analysis_duration_ms,
                metadata={
                    "analysis_mode": analysis_mode,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

            return self._apply_analysis_mode_expectation(
                result,
                case,
                analysis_mode,
            )

        # ------------------------------------------------------
        # 6. Registrar metadados operacionais e consumo de LLM.
        # ------------------------------------------------------

        evaluation_result.metadata.update(
            {
                "category": case.category,
                "analysis_mode": analysis_mode,
                "reference_sql_ms": reference_sql_ms,
                "agent_analysis_id": analysis.get("analysis_id"),
                "agent_total_results": analysis.get("total_results"),
                "agent_step_count": len(
                    analysis.get("steps", [])
                    if isinstance(analysis.get("steps"), list)
                    else []
                ),
                "agent_partial_success": analysis.get(
                    "partial_success",
                    False,
                ),
                "llm_metrics": llm_usage,
                **extraction_metadata,
            }
        )

        logger.info(
            "Cenário de avaliação concluído",
            extra={
                "event": "evaluation_case_completed",
                "case_id": case.case_id,
                "status": evaluation_result.status,
                "score": evaluation_result.score,
                "provider": actual_provider,
                "model": actual_model,
                "duration_ms": analysis_duration_ms,
                "llm_requests": llm_usage["requests"]["total"],
                "llm_tokens": llm_usage["tokens"]["total"],
            },
        )

        # ------------------------------------------------------
        # 7. Verificar se o modo executado corresponde
        #    à expectativa definida no cenário.
        # ------------------------------------------------------

        return self._apply_analysis_mode_expectation(
            evaluation_result,
            case,
            analysis_mode,
        )

    # ==========================================================
    # EXECUÇÃO COMPLETA
    # ==========================================================

    def run(
        self,
        cases: Iterable[EvaluationCase] | None = None,
        run_id: str | None = None,
    ) -> EvaluationRunReport:
        """Executa todos os cenários de uma avaliação.

        Quando cases não é informado, carrega o dataset padrão
        ou o arquivo especificado na inicialização.

        Cada cenário é executado independentemente. Uma falha
        individual não interrompe os próximos cenários.

        Returns:
            Relatório consolidado com resultados e métricas.
        """

        evaluation_cases = (
            list(cases)
            if cases is not None
            else load_evaluation_dataset(self.dataset_path)
        )

        if not all(isinstance(case, EvaluationCase) for case in evaluation_cases):
            raise TypeError("Todos os cenários devem ser EvaluationCase.")

        report = EvaluationRunReport(
            run_id=run_id or str(uuid4()),
            provider=self.provider_name,
            metadata={
                "dataset_path": (
                    str(self.dataset_path) if self.dataset_path is not None else None
                ),
                "scenario_count": len(evaluation_cases),
            },
        )

        if not evaluation_cases:
            return self._finalize_report(report)

        # Uma instância do agente por execução completa.
        # Isso permite reutilizar o cliente de IA dentro da execução,
        # sem reutilizar inadvertidamente o cache de uma execução
        # anterior do runner.

        try:
            analyst = self._create_analyst()

        except Exception as exc:
            logger.exception(
                "Falha ao inicializar BusinessAnalyst",
                extra={
                    "event": "evaluation_analyst_initialization_failed",
                    "provider": self.provider_name,
                },
            )

            for case in evaluation_cases:
                report.results.append(
                    self._error_result(
                        case,
                        exc,
                        phase="analyst_initialization",
                        provider=self.provider_name,
                    )
                )

            return self._finalize_report(report)

        report.model = getattr(
            analyst,
            "model",
            None,
        )

        logger.info(
            "Execução do AI Evaluation Engine iniciada",
            extra={
                "event": "evaluation_run_started",
                "run_id": report.run_id,
                "provider": self.provider_name,
                "model": report.model,
                "scenario_count": len(evaluation_cases),
            },
        )

        for index, case in enumerate(
            evaluation_cases,
            start=1,
        ):
            logger.info(
                "Executando cenário de avaliação",
                extra={
                    "event": "evaluation_case_started",
                    "run_id": report.run_id,
                    "case_id": case.case_id,
                    "position": index,
                    "total": len(evaluation_cases),
                },
            )

            result = self._run_case(
                case=case,
                analyst=analyst,
            )

            report.results.append(result)

        report.model = getattr(analyst, "model", None) or report.model

        self._finalize_report(report)

        logger.info(
            "Execução do AI Evaluation Engine finalizada",
            extra={
                "event": "evaluation_run_completed",
                "run_id": report.run_id,
                "provider": report.provider,
                "model": report.model,
                "total_cases": report.total_cases,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
                "error_cases": report.error_cases,
                "skipped_cases": report.skipped_cases,
                "pass_rate": report.pass_rate,
                "duration_ms": report.duration_ms,
            },
        )

        return report
