"""Orquestração das avaliações do AI Business Analyst."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
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

    Cenários SQL:
    - Executam o SQL de referência pelo SQLExecutor.
    - Executam a pergunta no BusinessAnalyst.
    - Comparam os resultados obtidos com a referência.
    - Validam a fundamentação das afirmações da resposta.

    Cenários comportamentais:
    - Não executam SQL de referência.
    - Executam a pergunta no BusinessAnalyst.
    - Avaliam o comportamento retornado pelo agente.

    O runner aceita dependências injetáveis para permitir testes
    sem chamadas reais aos provedores de IA.

    Cada cenário possui um contexto independente de métricas LLM,
    permitindo rastrear requisições, tokens, retries, latência
    e estimativas de custos por modelo.
    """

    def __init__(
        self,
        provider_name: str = "gemini",
        dataset_path: str | Path | None = None,
        analyst_factory: Callable[..., Any] | None = None,
        reference_executor: SQLExecutor | None = None,
        evaluator: ResultEvaluator | None = None,
    ):
        if not isinstance(provider_name, str):
            raise TypeError("provider_name deve ser uma string.")

        if not provider_name.strip():
            raise ValueError("provider_name deve ser uma string não vazia.")

        normalized_provider = provider_name.strip().lower()

        if normalized_provider not in SUPPORTED_PROVIDERS:
            available = ", ".join(sorted(SUPPORTED_PROVIDERS))

            raise ValueError(
                f"Provedor não suportado: '{provider_name}'. "
                f"Provedores disponíveis: {available}."
            )

        self.provider_name = normalized_provider

        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        self.analyst_factory = (
            analyst_factory if analyst_factory is not None else BusinessAnalyst
        )

        self.reference_executor = (
            reference_executor if reference_executor is not None else SQLExecutor()
        )

        self.evaluator = evaluator if evaluator is not None else ResultEvaluator()

    # ==========================================================
    # INICIALIZAÇÃO
    # ==========================================================

    def _create_analyst(self) -> Any:
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

        snapshot_method = getattr(analysis_metrics, "snapshot", None)

        if not callable(snapshot_method):
            raise TypeError("O contexto de métricas LLM não disponibiliza snapshot().")

        snapshot = snapshot_method()

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

    @staticmethod
    def _resolve_analysis_identity(
        analysis: Mapping[str, Any],
        *,
        default_provider: str,
        default_model: str | None,
        default_analysis_id: str,
    ) -> tuple[str, str | None, str]:
        """Valida e normaliza a identidade retornada pelo agente.

        O provedor precisa pertencer à lista suportada.
        Modelo e identificador precisam ser strings quando informados.

        Valores ausentes utilizam os valores de referência do runner
        e do contexto individual da avaliação.
        """

        raw_provider = analysis.get("provider")

        if raw_provider is None:
            raw_provider = default_provider

        elif isinstance(raw_provider, str) and not raw_provider.strip():
            raw_provider = default_provider

        if not isinstance(raw_provider, str):
            raise TypeError("O provedor retornado pelo agente deve ser uma string.")

        provider = raw_provider.strip().lower()

        if provider not in SUPPORTED_PROVIDERS:
            available = ", ".join(sorted(SUPPORTED_PROVIDERS))

            raise ValueError(
                f"Provedor retornado pelo agente não suportado: '{provider}'. "
                f"Provedores disponíveis: {available}."
            )

        raw_model = analysis.get("model")

        if raw_model is None:
            raw_model = default_model

        elif isinstance(raw_model, str) and not raw_model.strip():
            raw_model = default_model

        if raw_model is not None and not isinstance(raw_model, str):
            raise TypeError("O modelo retornado pelo agente deve ser uma string.")

        model = raw_model.strip() if isinstance(raw_model, str) else None

        raw_analysis_id = analysis.get("analysis_id")

        if raw_analysis_id is None:
            raw_analysis_id = default_analysis_id

        elif isinstance(raw_analysis_id, str) and not raw_analysis_id.strip():
            raw_analysis_id = default_analysis_id

        if not isinstance(raw_analysis_id, str):
            raise TypeError(
                "O identificador da análise retornado pelo agente deve ser uma string."
            )

        analysis_id = raw_analysis_id.strip()

        if not analysis_id:
            raise ValueError("O identificador da análise não pode ser vazio.")

        return provider, model, analysis_id

    @staticmethod
    def _extract_token_cost_estimate(
        llm_usage: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Extrai e copia a estimativa de custos do snapshot de LLM.

        O snapshot completo continua disponível em metadata["llm_metrics"].

        A cópia independente permite armazenar a estimativa diretamente
        no EvaluationCaseResult sem compartilhar referências mutáveis.
        """

        if not isinstance(llm_usage, Mapping):
            return None

        estimate = llm_usage.get("token_cost_estimate")

        if not isinstance(estimate, Mapping):
            return None

        return deepcopy(dict(estimate))

    @staticmethod
    def _get_llm_usage_count(
        llm_usage: Mapping[str, Any],
        section: str,
        key: str,
    ) -> int:
        """Obtém uma contagem de métricas sem presumir campos existentes."""

        values = llm_usage.get(section, {})

        if not isinstance(values, Mapping):
            return 0

        value = values.get(key, 0)

        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0

        return value

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
        Não utiliza o campo raiz 'results' como fallback para
        consultas compostas sem resultados individuais.
        """

        if not isinstance(queries, list):
            raise TypeError("As consultas de uma análise composta devem ser uma lista.")

        for index, query in enumerate(queries):
            if not isinstance(query, Mapping):
                raise TypeError(
                    f"A consulta {index} de uma análise composta "
                    "deve ser um objeto de mapeamento."
                )

        successful_queries = [
            query for query in queries if query.get("status") == "success"
        ]

        failed_queries = [
            query
            for query in queries
            if (
                query.get("status") == "failure"
                and query.get("failure_type") != "unsupported_capability"
            )
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

        if expected_mode not in SUPPORTED_ANALYSIS_MODES:
            raise ValueError(
                f"Modo de análise esperado não suportado: '{expected_mode}'."
            )

        if actual_mode not in SUPPORTED_ANALYSIS_MODES:
            raise ValueError(
                f"Modo de análise executado não suportado: '{actual_mode}'."
            )

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
    # VALIDAÇÃO DAS EVIDÊNCIAS DE RESPOSTAS SQL
    # ==========================================================

    def _validate_sql_answer_evidence(
        self,
        analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Valida as evidências textuais de uma análise SQL.

        A validação é independente da comparação entre resultados SQL.

        Análises legadas sem answer_evidence recebem not_evaluated.
        Erros internos de validação são registrados separadamente.
        """

        validator = getattr(
            self.evaluator,
            "validate_answer_evidence",
            None,
        )

        if not callable(validator):
            return {
                "status": "not_evaluated",
                "claims_count": 0,
                "valid_claims": 0,
                "invalid_claims": 0,
                "reference_count": 0,
                "valid_references": 0,
                "invalid_references": 0,
                "claims_supported": None,
                "answer_numeric_values": [],
                "unsupported_numeric_values": [],
                "uncovered_numeric_values": [],
                "differences": [
                    "O avaliador não disponibiliza validação de evidências."
                ],
            }

        try:
            report = validator(dict(analysis))

            if not isinstance(report, Mapping):
                raise TypeError(
                    "A validação de evidências deve retornar um mapeamento."
                )

            normalized_report = deepcopy(dict(report))

            if normalized_report.get("status") not in {
                "verified",
                "failed",
                "not_evaluated",
            }:
                raise ValueError("A validação retornou um status desconhecido.")

            claims_supported = normalized_report.get("claims_supported")

            if claims_supported is not None and not isinstance(
                claims_supported,
                bool,
            ):
                raise TypeError("claims_supported deve ser um booleano ou None.")

            counter_fields = (
                "claims_count",
                "valid_claims",
                "invalid_claims",
                "reference_count",
                "valid_references",
                "invalid_references",
            )

            for field in counter_fields:
                if field in normalized_report:
                    value = normalized_report[field]

                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError(
                            f"O contador '{field}' possui um valor inválido."
                        )

            if all(field in normalized_report for field in counter_fields[:3]):
                if (
                    normalized_report["valid_claims"]
                    + normalized_report["invalid_claims"]
                    > normalized_report["claims_count"]
                ):
                    raise ValueError(
                        "Os contadores de afirmações excedem o total informado."
                    )

            if all(field in normalized_report for field in counter_fields[3:]):
                if (
                    normalized_report["valid_references"]
                    + normalized_report["invalid_references"]
                    > normalized_report["reference_count"]
                ):
                    raise ValueError(
                        "Os contadores de referências excedem o total informado."
                    )

            raw_differences = normalized_report.get("differences", [])

            if isinstance(raw_differences, list):
                normalized_report["differences"] = [
                    difference.strip()
                    for difference in raw_differences
                    if isinstance(difference, str) and difference.strip()
                ]
            else:
                normalized_report["differences"] = []

            return normalized_report

        except Exception as exc:
            logger.exception(
                "Falha na validação das evidências da resposta SQL",
                extra={
                    "event": "evaluation_answer_evidence_failed",
                    "analysis_id": analysis.get("analysis_id"),
                },
            )

            return {
                "status": "error",
                "claims_count": 0,
                "valid_claims": 0,
                "invalid_claims": 0,
                "reference_count": 0,
                "valid_references": 0,
                "invalid_references": 0,
                "claims_supported": False,
                "answer_numeric_values": [],
                "unsupported_numeric_values": [],
                "uncovered_numeric_values": [],
                "error_type": type(exc).__name__,
                "differences": [
                    (
                        "Erro interno durante a validação das evidências: "
                        f"{type(exc).__name__}: {exc}"
                    )
                ],
            }

    @staticmethod
    def _attach_sql_answer_evidence(
        result: EvaluationCaseResult,
        evidence_report: Mapping[str, Any],
    ) -> EvaluationCaseResult:
        """Anexa a validação das afirmações ao resultado SQL.

        Comportamento:
        - verified: preserva o resultado original.
        - not_evaluated: mantém compatibilidade com análises legadas.
        - failed: reprova o cenário sem apagar o score SQL.
        - error: registra uma falha operacional de validação.

        A pontuação original continua representando exclusivamente
        a comparação entre os resultados SQL esperados e obtidos.
        """

        report = deepcopy(dict(evidence_report))

        evidence_status = report.get("status")

        # Normaliza diferenças mesmo quando o relatório veio de
        # uma implementação de avaliador alternativa.
        raw_differences = report.get("differences", [])

        if isinstance(raw_differences, list):
            differences = [
                difference.strip()
                for difference in raw_differences
                if isinstance(difference, str) and difference.strip()
            ]
        else:
            differences = []

        report["differences"] = differences

        result.metadata.update(
            {
                "answer_evidence_validation": report,
                "answer_evidence_status": evidence_status,
                "answer_claims_supported": report.get("claims_supported"),
            }
        )

        # ------------------------------------------------------
        # EVIDÊNCIAS INVÁLIDAS
        # ------------------------------------------------------

        if evidence_status == "failed":
            result.metadata.setdefault(
                "sql_result_status_before_evidence",
                result.status,
            )

            result.metadata.setdefault(
                "sql_result_score_before_evidence",
                result.score,
            )

            if result.status != "error":
                result.status = "failed"

            for difference in differences:
                message = f"Fundamentação da resposta: {difference}"

                if message not in result.differences:
                    result.differences.append(message)

            if not differences:
                message = (
                    "Fundamentação da resposta: "
                    "as evidências não passaram na validação determinística."
                )

                if message not in result.differences:
                    result.differences.append(message)

        # ------------------------------------------------------
        # ERRO INTERNO DE VALIDAÇÃO
        # ------------------------------------------------------

        elif evidence_status == "error":
            if result.status != "error":
                result.metadata.setdefault(
                    "sql_result_status_before_evidence",
                    result.status,
                )

                result.metadata.setdefault(
                    "sql_result_score_before_evidence",
                    result.score,
                )

                result.status = "error"

            result.error = (
                result.error or "Falha operacional durante a validação das evidências."
            )

        # ------------------------------------------------------
        # EVIDÊNCIAS VÁLIDAS OU NÃO AVALIADAS
        # ------------------------------------------------------

        return result

    # ==========================================================
    # CONTEXTO DO RESULTADO COMPORTAMENTAL
    # ==========================================================

    @staticmethod
    def _attach_behavior_metadata(
        result: EvaluationCaseResult,
        *,
        case: EvaluationCase,
        analysis: Mapping[str, Any],
        provider: str,
        model: str | None,
        analysis_id: str,
        duration_ms: float,
        llm_usage: dict[str, Any],
    ) -> EvaluationCaseResult:
        """Anexa contexto operacional e evidências comportamentais.

        Preserva:
        - Identificação do comportamento observado.
        - Justificativa e pergunta de esclarecimento.
        - Limitações e informações sustentadas.
        - Indicador de execução SQL.
        - Resposta original do agente.
        - Resultado individual de cada critério.
        - Contagem e percentual de critérios aprovados.

        O snapshot de métricas LLM permanece disponível nos metadados.
        A estimativa de custos é transferida para o campo próprio
        do EvaluationCaseResult.
        """

        raw_behavior = analysis.get("behavior")

        behavior = raw_behavior if isinstance(raw_behavior, Mapping) else {}

        observed_behavior = behavior.get("type")

        # ------------------------------------------------------
        # NORMALIZAÇÃO DAS EVIDÊNCIAS
        # ------------------------------------------------------

        def normalize_text(value: Any) -> str:
            """Normaliza campos textuais sem alterar seu conteúdo."""

            return value.strip() if isinstance(value, str) else ""

        raw_limitations = behavior.get("limitations")

        limitations = (
            deepcopy(raw_limitations) if isinstance(raw_limitations, list) else []
        )

        raw_supported_information = behavior.get("supported_information")

        supported_information = (
            deepcopy(raw_supported_information)
            if isinstance(raw_supported_information, list)
            else []
        )

        behavior_evidence = {
            "type": observed_behavior,
            "reason": normalize_text(behavior.get("reason")),
            "clarification_question": normalize_text(
                behavior.get("clarification_question")
            ),
            "limitations": limitations,
            "supported_information": supported_information,
            "sql_executed": behavior.get("sql_executed") is True,
        }

        # ------------------------------------------------------
        # RESPOSTA ORIGINAL
        # ------------------------------------------------------

        answer = analysis.get("answer")

        # ------------------------------------------------------
        # RESULTADOS DOS CRITÉRIOS COMPORTAMENTAIS
        # ------------------------------------------------------

        raw_criteria_results = result.metadata.get(
            "criteria_results",
            {},
        )

        behavior_checks = (
            deepcopy(dict(raw_criteria_results))
            if isinstance(raw_criteria_results, Mapping)
            else {}
        )

        passed_checks = sum(check is True for check in behavior_checks.values())

        total_checks = len(behavior_checks)

        behavior_pass_rate = (
            round(
                (passed_checks / total_checks) * 100,
                2,
            )
            if total_checks
            else 0.0
        )

        # ------------------------------------------------------
        # ALIASES DE EVIDÊNCIAS
        # ------------------------------------------------------

        raw_answer_evidence = result.metadata.get(
            "answer_evidence_validation",
            analysis.get("answer_evidence_validation"),
        )

        answer_evidence_status = result.metadata.get("answer_evidence_status")
        answer_claims_supported = result.metadata.get("answer_claims_supported")

        if isinstance(raw_answer_evidence, Mapping):
            answer_evidence_status = raw_answer_evidence.get(
                "status",
                answer_evidence_status,
            )

            answer_claims_supported = raw_answer_evidence.get(
                "claims_supported",
                answer_claims_supported,
            )

        if answer_claims_supported is not None and not isinstance(
            answer_claims_supported,
            bool,
        ):
            answer_claims_supported = None

        # ------------------------------------------------------
        # RASTREABILIDADE OPERACIONAL
        # ------------------------------------------------------

        result_fields = {
            "analysis_id": analysis_id,
            "provider": provider,
            "model": model,
            "duration_ms": duration_ms,
            "cache_hit": analysis.get("cache_hit"),
        }

        for field_name, value in result_fields.items():
            if hasattr(result, field_name):
                setattr(
                    result,
                    field_name,
                    value,
                )

        result.token_cost_estimate = EvaluationRunner._extract_token_cost_estimate(
            llm_usage
        )

        # ------------------------------------------------------
        # METADADOS FINAIS
        # ------------------------------------------------------

        result.metadata.update(
            {
                "category": case.category,
                "evaluation_type": case.evaluation_type,
                "expected_behavior": case.expected_behavior,
                "observed_behavior": observed_behavior,
                # Evidências originais do comportamento.
                "behavior_evidence": behavior_evidence,
                # Resposta textual original.
                "answer": answer,
                # Resultado individual dos critérios.
                "behavior_checks": behavior_checks,
                # Consolidação dos critérios.
                "passed_checks": passed_checks,
                "total_checks": total_checks,
                "behavior_pass_rate": behavior_pass_rate,
                # Aliases padronizados de evidências.
                "answer_evidence_status": answer_evidence_status,
                "answer_claims_supported": answer_claims_supported,
                # Cenários comportamentais não executam
                # SQL de referência.
                "reference_sql_executed": False,
                "reference_sql_ms": None,
                # Modo de análise.
                "analysis_mode": analysis.get(
                    "analysis_mode",
                    "single_query",
                ),
                # Identificação da análise.
                "agent_analysis_id": analysis.get("analysis_id") or analysis_id,
                # Resultados e execução.
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
                # Métricas individuais do provedor.
                "llm_metrics": llm_usage,
            }
        )

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
        """Monta um resultado de avaliação com falha operacional.

        Quando as métricas LLM já foram coletadas, preserva também
        a estimativa de custos no campo próprio do resultado.

        Campos essenciais do erro não podem ser sobrescritos
        por metadados adicionais.
        """

        result_metadata = {}

        if metadata:
            result_metadata.update(metadata)

        # Campos controlados pelo runner são aplicados por último
        # para impedir sobrescrita acidental ou externa.
        result_metadata.update(
            {
                "category": case.category,
                "evaluation_type": case.evaluation_type,
                "phase": phase,
                "error_type": type(error).__name__,
            }
        )

        if case.expected_analysis_mode is not None:
            result_metadata["expected_analysis_mode"] = case.expected_analysis_mode

        if case.expected_behavior is not None:
            result_metadata["expected_behavior"] = case.expected_behavior

        llm_usage = result_metadata.get("llm_metrics")

        token_cost_estimate = (
            EvaluationRunner._extract_token_cost_estimate(llm_usage)
            if isinstance(llm_usage, Mapping)
            else None
        )

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
            token_cost_estimate=token_cost_estimate,
        )

    # ==========================================================
    # FINALIZAÇÃO E MÉTRICAS
    # ==========================================================

    @staticmethod
    def _finalize_report(
        report: EvaluationRunReport,
    ) -> EvaluationRunReport:
        """Calcula métricas e finaliza o relatório.

        O resumo de custos é calculado a partir dos resultados
        individuais por EvaluationRunReport.token_cost_summary.
        """

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
        """Executa um cenário SQL ou comportamental."""

        is_behavioral_case = case.evaluation_type != "sql_result"

        reference_sql_ms: float | None = None
        expected_rows: list[dict[str, Any]] | None = None

        # ------------------------------------------------------
        # 1. Calcular o resultado de referência.
        #
        # Somente cenários SQL executam a referência.
        # Cenários comportamentais não possuem SQL esperado.
        # ------------------------------------------------------

        if not is_behavioral_case:
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
                    metadata={
                        "reference_sql_executed": True,
                    },
                )

        # ------------------------------------------------------
        # 2. Executar a pergunta no agente.
        #
        # Cada cenário recebe um contexto independente.
        # Isso impede que tokens, custos e latências de cenários
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
                "reference_sql_executed": not is_behavioral_case,
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
                    "reference_sql_executed": not is_behavioral_case,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

        # ------------------------------------------------------
        # 4. Validar o modo de análise retornado pelo agente.
        # ------------------------------------------------------

        analysis_mode = analysis.get(
            "analysis_mode",
            "single_query",
        )

        if (
            not isinstance(analysis_mode, str)
            or analysis_mode not in SUPPORTED_ANALYSIS_MODES
        ):
            error = ValueError(
                f"Modo de análise retornado pelo agente não suportado: "
                f"'{analysis_mode}'."
            )

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
                    "analysis_mode": analysis_mode,
                    "reference_sql_executed": not is_behavioral_case,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

        # ------------------------------------------------------
        # 5. Validar a identidade retornada pelo agente.
        # ------------------------------------------------------

        try:
            actual_provider, actual_model, actual_analysis_id = (
                self._resolve_analysis_identity(
                    analysis,
                    default_provider=self.provider_name,
                    default_model=getattr(analyst, "model", None),
                    default_analysis_id=analysis_tracking_id,
                )
            )

        except (TypeError, ValueError) as exc:
            llm_usage = self._snapshot_llm_metrics(
                analysis_metrics,
                provider=self.provider_name,
                model=getattr(analyst, "model", None),
                analysis_id=analysis_tracking_id,
            )

            return self._error_result(
                case,
                exc,
                phase="analysis_validation",
                provider=self.provider_name,
                model=getattr(analyst, "model", None),
                duration_ms=analysis_duration_ms,
                metadata={
                    "analysis_mode": analysis_mode,
                    "reported_provider": analysis.get("provider"),
                    "reported_model": analysis.get("model"),
                    "reported_analysis_id": analysis.get("analysis_id"),
                    "reference_sql_executed": not is_behavioral_case,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

        llm_usage = self._snapshot_llm_metrics(
            analysis_metrics,
            provider=actual_provider,
            model=actual_model,
            analysis_id=actual_analysis_id,
        )

        # Extrai uma cópia independente do custo para o resultado.
        token_cost_estimate = self._extract_token_cost_estimate(llm_usage)

        # ------------------------------------------------------
        # 6. Avaliação comportamental.
        #
        # Este caminho é independente da comparação SQL.
        #
        # Não executa SQL de referência.
        # Não extrai resultados tabulares.
        # Não aplica expectativas single_query/multi_query.
        # ------------------------------------------------------

        if is_behavioral_case:
            try:
                evaluation_result = self.evaluator.evaluate_behavior(
                    case,
                    analysis,
                )

            except Exception as exc:
                logger.exception(
                    "Falha durante a avaliação comportamental",
                    extra={
                        "event": "evaluation_behavior_evaluation_failed",
                        "case_id": case.case_id,
                        "provider": actual_provider,
                        "model": actual_model,
                    },
                )

                return self._error_result(
                    case,
                    exc,
                    phase="behavior_evaluation",
                    provider=actual_provider,
                    model=actual_model,
                    duration_ms=analysis_duration_ms,
                    metadata={
                        "reference_sql_executed": False,
                        "reference_sql_ms": None,
                        "analysis_mode": analysis_mode,
                        "agent_analysis_id": actual_analysis_id,
                        "llm_metrics": llm_usage,
                    },
                )

            evaluation_result = self._attach_behavior_metadata(
                evaluation_result,
                case=case,
                analysis=analysis,
                provider=actual_provider,
                model=actual_model,
                analysis_id=actual_analysis_id,
                duration_ms=analysis_duration_ms,
                llm_usage=llm_usage,
            )

            logger.info(
                "Cenário comportamental concluído",
                extra={
                    "event": "evaluation_behavior_case_completed",
                    "case_id": case.case_id,
                    "evaluation_type": case.evaluation_type,
                    "expected_behavior": case.expected_behavior,
                    "observed_behavior": (
                        evaluation_result.metadata.get("observed_behavior")
                    ),
                    "status": evaluation_result.status,
                    "score": evaluation_result.score,
                    "provider": actual_provider,
                    "model": actual_model,
                    "duration_ms": analysis_duration_ms,
                    "llm_requests": self._get_llm_usage_count(
                        llm_usage,
                        "requests",
                        "total",
                    ),
                    "llm_tokens": self._get_llm_usage_count(
                        llm_usage,
                        "tokens",
                        "total",
                    ),
                    "estimated_cost_usd": evaluation_result.estimated_cost_usd,
                },
            )

            return evaluation_result

        # A partir daqui, o caminho é exclusivamente SQL.
        # expected_rows foi calculado no início do método.
        assert expected_rows is not None

        # Valida a fundamentação textual independentemente
        # da comparação dos resultados SQL.
        answer_evidence_validation = self._validate_sql_answer_evidence(analysis)

        extraction_metadata: dict[str, Any] = {}

        # ------------------------------------------------------
        # 7. Extrair resultados simples ou compostos.
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
                            "evaluation_type": case.evaluation_type,
                            "analysis_mode": analysis_mode,
                            "reference_sql_executed": True,
                            "reference_sql_ms": reference_sql_ms,
                            "llm_metrics": llm_usage,
                            **extraction_metadata,
                        },
                        token_cost_estimate=token_cost_estimate,
                    )

                    result = self._apply_analysis_mode_expectation(
                        result,
                        case,
                        analysis_mode,
                    )

                    result = self._attach_sql_answer_evidence(
                        result,
                        answer_evidence_validation,
                    )

                    logger.info(
                        "Cenário não pôde ser comparado",
                        extra={
                            "event": "evaluation_case_not_comparable",
                            "case_id": case.case_id,
                            "status": result.status,
                            "reason": reason,
                            "provider": actual_provider,
                            "model": actual_model,
                            "estimated_cost_usd": result.estimated_cost_usd,
                        },
                    )

                    return result

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
                    "reference_sql_executed": True,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

            result = self._apply_analysis_mode_expectation(
                result,
                case,
                analysis_mode,
            )

            return self._attach_sql_answer_evidence(
                result,
                answer_evidence_validation,
            )

        # ------------------------------------------------------
        # 8. Comparar resultado real com referência.
        # ------------------------------------------------------

        try:
            evaluation_result = self.evaluator.evaluate(
                case=case,
                expected_rows=expected_rows,
                actual_rows=actual_rows,
                analysis_id=actual_analysis_id,
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
                    "reference_sql_executed": True,
                    "reference_sql_ms": reference_sql_ms,
                    "llm_metrics": llm_usage,
                },
            )

            result = self._apply_analysis_mode_expectation(
                result,
                case,
                analysis_mode,
            )

            return self._attach_sql_answer_evidence(
                result,
                answer_evidence_validation,
            )

        # ------------------------------------------------------
        # 9. Registrar metadados operacionais e consumo de LLM.
        # ------------------------------------------------------

        evaluation_result.token_cost_estimate = token_cost_estimate

        evaluation_result.metadata.update(
            {
                "category": case.category,
                "evaluation_type": case.evaluation_type,
                "reference_sql_executed": True,
                "analysis_mode": analysis_mode,
                "reference_sql_ms": reference_sql_ms,
                "agent_analysis_id": actual_analysis_id,
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

        # ------------------------------------------------------
        # 10. Verificar o modo de análise esperado.
        # ------------------------------------------------------

        evaluation_result = self._apply_analysis_mode_expectation(
            evaluation_result,
            case,
            analysis_mode,
        )

        # ------------------------------------------------------
        # 11. Validar e anexar as evidências da resposta.
        # ------------------------------------------------------

        evaluation_result = self._attach_sql_answer_evidence(
            evaluation_result,
            answer_evidence_validation,
        )

        # ------------------------------------------------------
        # 12. Registrar resultado final.
        # ------------------------------------------------------

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
                "llm_requests": self._get_llm_usage_count(
                    llm_usage,
                    "requests",
                    "total",
                ),
                "llm_tokens": self._get_llm_usage_count(
                    llm_usage,
                    "tokens",
                    "total",
                ),
                "estimated_cost_usd": evaluation_result.estimated_cost_usd,
                "answer_evidence_status": (
                    evaluation_result.metadata.get("answer_evidence_status")
                ),
                "answer_claims_supported": (
                    evaluation_result.metadata.get("answer_claims_supported")
                ),
            },
        )

        return evaluation_result

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

        # Valida as expectativas antes de iniciar chamadas ao agente.
        # Um dataset inválido deve falhar explicitamente, sem produzir
        # resultados enganosos de divergência de modo.
        for case in evaluation_cases:
            expected_mode = case.expected_analysis_mode

            if expected_mode is not None and (
                not isinstance(expected_mode, str)
                or expected_mode not in SUPPORTED_ANALYSIS_MODES
            ):
                raise ValueError(
                    f"O cenário '{case.case_id}' possui um modo de análise "
                    f"esperado não suportado: '{expected_mode}'."
                )

        report = EvaluationRunReport(
            run_id=run_id or str(uuid4()),
            provider=self.provider_name,
            metadata={
                "dataset_path": (
                    str(self.dataset_path) if self.dataset_path is not None else None
                ),
                "scenario_count": len(evaluation_cases),
                "sql_scenario_count": sum(
                    case.evaluation_type == "sql_result" for case in evaluation_cases
                ),
                "behavioral_scenario_count": sum(
                    case.evaluation_type != "sql_result" for case in evaluation_cases
                ),
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
                "sql_scenario_count": report.metadata["sql_scenario_count"],
                "behavioral_scenario_count": report.metadata[
                    "behavioral_scenario_count"
                ],
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
                    "evaluation_type": case.evaluation_type,
                    "position": index,
                    "total": len(evaluation_cases),
                },
            )

            try:
                result = self._run_case(
                    case=case,
                    analyst=analyst,
                )

            except Exception as exc:
                # Última barreira de isolamento:
                # uma exceção inesperada em um cenário não pode
                # interromper a execução dos cenários seguintes.
                logger.exception(
                    "Falha inesperada durante a execução do cenário",
                    extra={
                        "event": "evaluation_case_unhandled_exception",
                        "run_id": report.run_id,
                        "case_id": case.case_id,
                        "position": index,
                        "provider": self.provider_name,
                    },
                )

                result = self._error_result(
                    case,
                    exc,
                    phase="case_execution",
                    provider=self.provider_name,
                    model=report.model,
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
                "estimated_cost_usd": report.estimated_cost_usd,
                "cost_is_complete": report.token_cost_summary["is_complete"],
            },
        )

        return report
