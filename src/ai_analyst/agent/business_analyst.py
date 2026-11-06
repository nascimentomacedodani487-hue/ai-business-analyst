"""
Orquestrador principal do AI Business Analyst.
"""

import duckdb
import logging
import re
import unicodedata
from time import perf_counter

from ai_analyst.agent.sql_generator import SQLGenerator
from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.agent.response_generator import ResponseGenerator
from ai_analyst.agent.cache import AnalysisCache
from ai_analyst.data.db import get_database_version
from ai_analyst.llm.factory import create_llm_provider
from ai_analyst.security.guardrails import SQLSyntaxError


logger = logging.getLogger(__name__)


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

# Número máximo de correções semânticas solicitadas ao LLM.
MAX_SQL_CORRECTIONS = 2


class BusinessAnalyst:
    """Coordena a análise de perguntas de negócio."""

    def __init__(
        self,
        cache: AnalysisCache | None = None,
        provider_name: str = "gemini",
    ):
        normalized_provider = provider_name.strip().lower()

        if not normalized_provider:
            normalized_provider = "gemini"

        if normalized_provider not in DEFAULT_MODELS:
            available = ", ".join(DEFAULT_MODELS)

            raise ValueError(
                f"Provedor de IA não suportado: "
                f"'{normalized_provider}'. "
                f"Provedores disponíveis: {available}."
            )

        self.provider_name = normalized_provider
        self.model = DEFAULT_MODELS[normalized_provider]

        self.provider = None

        # Compatibilidade com código e testes anteriores.
        self.gemini = None

        self.sql_executor = SQLExecutor()
        self.cache = cache if cache is not None else AnalysisCache()

        self.sql_generator = None
        self.response_generator = None

        # Armazena as etapas da análise atual, inclusive em caso de falha.
        self.steps: list[dict] = []

        # Preserva o último SQL gerado, mesmo quando a análise falha.
        self.last_sql: str | None = None

    def _initialize_ai_components(self):
        """Inicializa os componentes usando o provedor selecionado."""

        if self.provider is None:

            # Compatibilidade com código antigo que define self.gemini.
            if self.gemini is not None:
                self.provider = self.gemini

            elif self.provider_name == "gemini":
                self.provider = create_llm_provider()

            else:
                self.provider = create_llm_provider(
                    self.provider_name
                )

        # Clientes simulados antigos podem não declarar o modelo.
        self.model = getattr(
            self.provider,
            "model",
            DEFAULT_MODELS[self.provider_name],
        )

        # Alias mantido para compatibilidade.
        self.gemini = self.provider

        # Inicializa os geradores caso ainda não existam.
        if self.sql_generator is None:
            self.sql_generator = SQLGenerator(
                provider=self.provider
            )

        if self.response_generator is None:
            self.response_generator = ResponseGenerator(
                provider=self.provider
            )

    @staticmethod
    def _is_numeric(value) -> bool:
        """Verifica se um valor é numérico, excluindo booleanos."""

        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normaliza texto para comparação sem acentos."""

        normalized = unicodedata.normalize("NFKD", str(value))

        return "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        ).lower()

    @classmethod
    def _tokenize(cls, value: str) -> set[str]:
        """Extrai palavras relevantes de um texto."""

        normalized = cls._normalize_text(value)

        return set(re.findall(r"[a-z0-9]+", normalized))

    @staticmethod
    def _humanize_column_name(column: str) -> str:
        """Transforma nomes de colunas em títulos legíveis."""

        return column.replace("_", " ").strip().title()

    @classmethod
    def _select_value_column(
        cls,
        question: str,
        numeric_columns: list[str],
    ) -> str:
        """Seleciona a métrica mais relevante para a pergunta."""

        if not numeric_columns:
            raise ValueError(
                "Nenhuma coluna numérica disponível para o gráfico."
            )

        question_tokens = cls._tokenize(question)

        metric_groups = {
            "cancellation_rate": {
                "question": {
                    "cancelamento", "cancelamentos",
                    "cancelado", "cancelados",
                    "cancelada", "canceladas",
                    "taxa", "percentual", "porcentagem",
                    "percent", "percentage", "rate",
                },
                "columns": {
                    "cancellation_rate",
                    "cancellation_rate_percentage",
                    "cancelation_rate",
                    "cancelation_rate_percentage",
                    "taxa_cancelamento",
                    "taxa_de_cancelamento",
                    "percentual_cancelamento",
                    "cancelamento_percentage",
                },
                "column_tokens": {
                    "cancellation", "cancelation",
                    "cancelamento", "cancelados",
                    "canceladas", "rate", "percentage",
                    "percentual", "taxa",
                },
            },
            "revenue": {
                "question": {
                    "faturamento", "receita", "vendas",
                    "venda", "financeiro", "arrecadacao",
                    "monetario",
                },
                "columns": {
                    "revenue", "total_revenue", "total_sales",
                    "sales", "faturamento", "receita",
                    "total_revenue_amount",
                },
                "column_tokens": {
                    "revenue", "sales", "sale",
                    "faturamento", "receita",
                    "amount", "valor", "total",
                },
            },
            "orders": {
                "question": {
                    "pedido", "pedidos", "ordem", "ordens",
                    "volume", "quantidade", "total",
                },
                "columns": {
                    "total_orders", "orders_count",
                    "order_count", "orders", "pedidos",
                    "quantidade_pedidos",
                },
                "column_tokens": {
                    "orders", "order", "pedidos",
                    "pedido", "count", "quantidade", "total",
                },
            },
            "canceled_orders": {
                "question": {
                    "cancelado", "cancelados",
                    "cancelada", "canceladas",
                    "cancelamento", "cancelamentos",
                },
                "columns": {
                    "canceled_orders", "cancelled_orders",
                    "cancelados", "pedidos_cancelados",
                    "total_canceled_orders",
                },
                "column_tokens": {
                    "canceled", "cancelled",
                    "cancelados", "canceladas",
                    "cancelamento",
                },
            },
            "delivered_orders": {
                "question": {
                    "entregue", "entregues",
                    "entregados", "entregadas",
                    "entrega", "entregas",
                },
                "columns": {
                    "delivered_orders", "orders_delivered",
                    "pedidos_entregues",
                    "total_delivered_orders",
                },
                "column_tokens": {
                    "delivered", "entregues",
                    "entregue", "entregados", "entregadas",
                },
            },
            "average": {
                "question": {
                    "media", "medio", "ticket",
                    "average", "avg", "promedio",
                },
                "columns": {
                    "average", "avg", "average_order_value",
                    "average_ticket", "ticket_medio", "media",
                },
                "column_tokens": {
                    "average", "avg", "media", "medio", "ticket",
                },
            },
            "percentage": {
                "question": {
                    "percentual", "porcentagem", "percent",
                    "percentage", "participacao",
                    "proporcao", "taxa", "share",
                },
                "columns": {
                    "percentage", "percent", "rate",
                    "percentage_rate", "share",
                    "proportion", "percentual", "taxa",
                },
                "column_tokens": {
                    "percentage", "percent", "rate",
                    "share", "proportion", "percentual", "taxa",
                },
            },
        }

        requested_groups = []

        for group_name, config in metric_groups.items():
            if question_tokens.intersection(config["question"]):
                requested_groups.append(group_name)

        if (
            "cancellation_rate" in requested_groups
            or (
                "canceled_orders" in requested_groups
                and "percentage" in requested_groups
            )
        ):
            requested_groups = ["cancellation_rate"]

        best_column = None
        best_score = 0

        for column in numeric_columns:
            normalized_column = cls._normalize_text(column)
            column_tokens = cls._tokenize(column)

            score = 0

            for group_name in requested_groups:
                config = metric_groups[group_name]

                if normalized_column in config["columns"]:
                    score += 100

                matching_tokens = column_tokens.intersection(
                    config["column_tokens"]
                )

                score += len(matching_tokens) * 10

                if group_name == "cancellation_rate":
                    if (
                        "rate" in column_tokens
                        or "taxa" in column_tokens
                    ):
                        score += 30

                    if (
                        "percentage" in column_tokens
                        or "percentual" in column_tokens
                    ):
                        score += 30

                    if (
                        "cancellation" in column_tokens
                        or "cancelation" in column_tokens
                        or "cancelamento" in column_tokens
                    ):
                        score += 30

            if score > best_score:
                best_score = score
                best_column = column

        if best_column is None or best_score == 0:
            return numeric_columns[0]

        logger.info(
            "Métrica selecionada para o gráfico",
            extra={
                "event": "chart_metric_selected",
                "question": question,
                "selected_column": best_column,
                "score": best_score,
            },
        )

        return best_column

    def _build_single_row_comparison(
        self,
        result: dict,
    ) -> dict | None:
        """Identifica comparações armazenadas em uma única linha."""

        if not isinstance(result, dict):
            return None

        columns = list(result.keys())

        comparison_pairs = [
            (
                "top_city",
                "top_city_revenue",
                "second_city",
                "second_city_revenue",
            ),
            (
                "first_city",
                "first_city_revenue",
                "second_city",
                "second_city_revenue",
            ),
            (
                "top_category",
                "top_category_revenue",
                "second_category",
                "second_category_revenue",
            ),
            (
                "first_category",
                "first_category_revenue",
                "second_category",
                "second_category_revenue",
            ),
            (
                "top_product",
                "top_product_revenue",
                "second_product",
                "second_product_revenue",
            ),
            (
                "first_product",
                "first_product_revenue",
                "second_product",
                "second_product_revenue",
            ),
        ]

        for (
            first_label_key,
            first_value_key,
            second_label_key,
            second_value_key,
        ) in comparison_pairs:

            required_columns = (
                first_label_key,
                first_value_key,
                second_label_key,
                second_value_key,
            )

            if not all(key in columns for key in required_columns):
                continue

            first_label = result.get(first_label_key)
            first_value = result.get(first_value_key)
            second_label = result.get(second_label_key)
            second_value = result.get(second_value_key)

            if (
                first_label is None
                or second_label is None
                or not self._is_numeric(first_value)
                or not self._is_numeric(second_value)
            ):
                continue

            label_name = (
                "city"
                if "city" in first_label_key
                else "category"
                if "category" in first_label_key
                else "product"
            )

            value_name = "revenue"

            chart_data = [
                {
                    label_name: str(first_label),
                    value_name: float(first_value),
                },
                {
                    label_name: str(second_label),
                    value_name: float(second_value),
                },
            ]

            return {
                "type": "bar",
                "title": (
                    f"{self._humanize_column_name(value_name)} "
                    f"por {self._humanize_column_name(label_name)}"
                ),
                "label_key": label_name,
                "value_key": value_name,
                "data": chart_data,
            }

        ranking_prefixes = ("top_", "first_", "second_", "third_")

        for label_key in columns:
            if not any(
                label_key.startswith(prefix)
                for prefix in ranking_prefixes
            ):
                continue

            suffix = label_key.split("_", 1)[1]
            label_value = result.get(label_key)

            if label_value is None:
                continue

            value_candidates = [
                f"{label_key}_value",
                f"{label_key}_revenue",
                f"{label_key}_total",
                f"{label_key}_count",
                f"{label_key}_amount",
            ]

            value_key = next(
                (
                    key
                    for key in value_candidates
                    if key in columns
                    and self._is_numeric(result.get(key))
                ),
                None,
            )

            if value_key is None:
                continue

            second_label_key = f"second_{suffix}"

            if second_label_key not in columns:
                continue

            second_label = result.get(second_label_key)

            if second_label is None:
                continue

            second_value_candidates = [
                f"{second_label_key}_value",
                f"{second_label_key}_revenue",
                f"{second_label_key}_total",
                f"{second_label_key}_count",
                f"{second_label_key}_amount",
            ]

            second_value_key = next(
                (
                    key
                    for key in second_value_candidates
                    if key in columns
                    and self._is_numeric(result.get(key))
                ),
                None,
            )

            if second_value_key is None:
                continue

            value_name = suffix

            chart_data = [
                {
                    "category": str(label_value),
                    value_name: float(result[value_key]),
                },
                {
                    "category": str(second_label),
                    value_name: float(result[second_value_key]),
                },
            ]

            return {
                "type": "bar",
                "title": (
                    f"{self._humanize_column_name(value_name)} "
                    "por categoria"
                ),
                "label_key": "category",
                "value_key": value_name,
                "data": chart_data,
            }

        return None

    def _build_chart_config(
        self,
        results: list[dict],
        question: str = "",
    ) -> dict | None:
        """Seleciona uma visualização com base na pergunta."""

        if not results:
            return None

        if len(results) == 1:
            comparison_chart = self._build_single_row_comparison(
                results[0]
            )

            if comparison_chart is not None:
                return comparison_chart

            return None

        columns = list(results[0].keys())

        if len(columns) < 2:
            return None

        numeric_columns = [
            column
            for column in columns
            if all(
                row.get(column) is not None
                and self._is_numeric(row.get(column))
                for row in results
            )
        ]

        categorical_columns = [
            column
            for column in columns
            if column not in numeric_columns
            and all(
                row.get(column) is not None
                for row in results
            )
        ]

        if not numeric_columns or not categorical_columns:
            return None

        label_key = max(
            categorical_columns,
            key=lambda column: len({
                str(row[column]) for row in results
            }),
        )

        categories = {
            str(row[label_key]) for row in results
        }

        if len(categories) > 20:
            return None

        value_key = self._select_value_column(
            question=question,
            numeric_columns=numeric_columns,
        )

        label_name = label_key.lower()

        time_terms = (
            "date", "month", "year", "day",
            "data", "mes", "ano", "dia",
        )

        proportion_terms = (
            "status", "category", "categoria",
            "type", "tipo",
        )

        if any(term in label_name for term in time_terms):
            chart_type = "line"

        elif (
            any(term in label_name for term in proportion_terms)
            and len(categories) <= 8
        ):
            chart_type = "pie"

        else:
            chart_type = "bar"

        chart_data = [
            {
                key: (
                    float(row[key])
                    if self._is_numeric(row.get(key))
                    else row.get(key)
                )
                for key in (label_key, value_key)
            }
            for row in results
        ]

        return {
            "type": chart_type,
            "title": (
                f"{self._humanize_column_name(value_key)} "
                f"por {self._humanize_column_name(label_key)}"
            ),
            "label_key": label_key,
            "value_key": value_key,
            "data": chart_data,
        }

    def analyze(self, question: str) -> dict:
        """Gera SQL, executa a consulta e interpreta os resultados."""

        # Reinicia os dados da análise atual.
        self.steps = []
        self.last_sql = None

        if not question or not question.strip():
            raise ValueError("A pergunta não pode estar vazia.")

        analysis_start = perf_counter()
        current_stage = "database_version"
        steps = self.steps

        def record_stage(name, status, duration_ms, details=None):
            step = {
                "name": name,
                "status": status,
                "duration_ms": duration_ms,
            }

            if details:
                step["details"] = details

            steps.append(step)

        def run_stage(name, operation):
            """Executa uma etapa e registra seu resultado e duração."""

            nonlocal current_stage

            current_stage = name
            stage_start = perf_counter()

            try:
                result = operation()

            except Exception:
                duration_ms = round(
                    (perf_counter() - stage_start) * 1000, 2
                )

                record_stage(name, "failed", duration_ms)
                raise

            duration_ms = round(
                (perf_counter() - stage_start) * 1000, 2
            )

            record_stage(name, "completed", duration_ms)

            return result

        try:
            # 1. Identificar a versão atual do banco.
            data_version = run_stage(
                "database_version",
                get_database_version,
            )

            # 2. Consultar cache isolado por provedor e modelo.
            cached_result = run_stage(
                "cache_lookup",
                lambda: self.cache.get(
                    question=question,
                    data_version=data_version,
                    provider=self.provider_name,
                    model=self.model,
                ),
            )

            if cached_result is not None:

                if (
                    cached_result.get("chart") is None
                    and cached_result.get("results")
                ):
                    chart = run_stage(
                        "chart_generation",
                        lambda: self._build_chart_config(
                            cached_result["results"],
                            question=question,
                        ),
                    )

                    cached_result["chart"] = chart

                    if chart is not None:
                        self.cache.set(
                            question=question,
                            data_version=data_version,
                            result=cached_result,
                            provider=self.provider_name,
                            model=self.model,
                        )

                cached_result["cache_hit"] = True
                cached_result["provider"] = self.provider_name
                cached_result["model"] = self.model

                total_time = round(
                    (perf_counter() - analysis_start) * 1000, 2
                )

                record_stage(
                    "cache_hit",
                    "completed",
                    0,
                    "Análise recuperada do cache. "
                    "Nenhuma chamada à IA foi realizada.",
                )

                cached_result["steps"] = steps
                cached_result["metrics"] = {
                    "total_time_ms": total_time,
                    "cache_hit": True,
                }

                logger.info(
                    "Análise recuperada do cache",
                    extra={
                        "event": "analysis_cache_hit",
                        "provider": self.provider_name,
                        "model": self.model,
                        "total_time_ms": total_time,
                    },
                )

                return cached_result

            logger.info(
                "Cache miss: iniciando nova análise",
                extra={
                    "event": "analysis_cache_miss",
                    "provider": self.provider_name,
                    "model": self.model,
                },
            )

            # 3. Inicializar componentes de IA.
            run_stage(
                "ai_initialization",
                self._initialize_ai_components,
            )

            # 4. Gerar SQL.
            sql_generation_start = perf_counter()
            pending_error = None
            sql = None

            try:
                sql = run_stage(
                    "sql_generation",
                    lambda: self.sql_generator.generate(question),
                )

                # Preserva o SQL assim que a geração termina.
                self.last_sql = sql

            except SQLSyntaxError as exc:
                pending_error = exc
                sql = exc.sql

                # Preserva também o SQL rejeitado pelos guardrails.
                self.last_sql = sql

            sql_generation_ms = round(
                (perf_counter() - sql_generation_start) * 1000, 2
            )

            # 5. Executar SQL com correção limitada.
            sql_execution_elapsed = 0.0
            correction_count = 0

            self.sql_executor.set_stage_callback(record_stage)

            try:
                while True:
                    if pending_error is not None:
                        if correction_count >= MAX_SQL_CORRECTIONS:
                            raise pending_error

                        correction_count += 1
                        correction_number = correction_count

                        failed_sql = getattr(
                            pending_error,
                            "sql",
                            sql,
                        )

                        error_message = str(pending_error)

                        logger.warning(
                            "Recoverable SQL error. Requesting correction.",
                            extra={
                                "event": "sql_correction_requested",
                                "provider": self.provider_name,
                                "model": self.model,
                                "correction_attempt": correction_number,
                                "max_corrections": MAX_SQL_CORRECTIONS,
                                "error_type": type(pending_error).__name__,
                            },
                        )

                        try:
                            sql = run_stage(
                                f"sql_correction_{correction_number}",
                                lambda failed_sql=failed_sql,
                                error_message=error_message:
                                    self.sql_generator.correct_sql(
                                        question=question,
                                        failed_sql=failed_sql,
                                        error_message=error_message,
                                    ),
                            )

                            # Atualiza para a versão corrigida.
                            self.last_sql = sql
                            pending_error = None

                        except SQLSyntaxError as exc:
                            pending_error = exc
                            sql = exc.sql

                            # Preserva o SQL retornado na correção,
                            # mesmo quando ele continua inválido.
                            self.last_sql = sql

                            continue

                    current_stage = (
                        f"sql_execution_attempt_{correction_count + 1}"
                    )

                    # Garante que o SQL da tentativa atual esteja salvo
                    # antes de iniciar a execução.
                    self.last_sql = sql

                    execution_start = perf_counter()

                    try:
                        results = self.sql_executor.execute(sql)
                        pending_error = None
                        break

                    except (
                        SQLSyntaxError,
                        duckdb.ProgrammingError,
                    ) as exc:
                        pending_error = exc

                        # Se a exceção carregar outro SQL, preserva-o.
                        self.last_sql = getattr(
                            exc,
                            "sql",
                            None,
                        ) or sql

                        sql = self.last_sql

                    finally:
                        sql_execution_elapsed += (
                            perf_counter() - execution_start
                        )

            finally:
                self.sql_executor.set_stage_callback(None)

            # Considera somente o tempo dentro das chamadas ao executor.
            # Correções solicitadas ao LLM ficam fora desta métrica.
            sql_execution_ms = round(
                sql_execution_elapsed * 1000,
                2,
            )

            # 6. Gerar interpretação.
            response_generation_start = perf_counter()

            answer = run_stage(
                "response_generation",
                lambda: self.response_generator.generate(
                    question=question,
                    sql=sql,
                    results=results,
                ),
            )

            response_generation_ms = round(
                (perf_counter() - response_generation_start) * 1000, 2
            )

            # 7. Gerar configuração do gráfico.
            chart = run_stage(
                "chart_generation",
                lambda: self._build_chart_config(
                    results,
                    question=question,
                ),
            )

            # 8. Calcular tempo total.
            total_time = round(
                (perf_counter() - analysis_start) * 1000, 2
            )

            # 9. Montar resultado.
            analysis_result = {
                "question": question,
                "sql": sql,
                "results": results,
                "total_results": len(results),
                "answer": answer,
                "chart": chart,
                "provider": self.provider_name,
                "model": self.model,
                "cache_hit": False,
                "steps": steps,
                "metrics": {
                    "total_time_ms": total_time,
                    "sql_generation_ms": sql_generation_ms,
                    "sql_execution_ms": sql_execution_ms,
                    "response_generation_ms": response_generation_ms,
                    "cache_hit": False,
                },
            }

            # 10. Armazenar no cache.
            current_stage = "cache_storage"

            self.cache.set(
                question=question,
                data_version=data_version,
                result=analysis_result,
                provider=self.provider_name,
                model=self.model,
            )

            logger.info(
                "Análise concluída com sucesso",
                extra={
                    "event": "analysis_completed",
                    "provider": self.provider_name,
                    "model": self.model,
                    "total_time_ms": total_time,
                    "sql_generation_ms": sql_generation_ms,
                    "sql_execution_ms": sql_execution_ms,
                    "response_generation_ms": response_generation_ms,
                    "total_results": len(results),
                    "cache_hit": False,
                },
            )

            return analysis_result

        except Exception:
            total_time = round(
                (perf_counter() - analysis_start) * 1000, 2
            )

            logger.exception(
                "Falha durante a análise de negócio",
                extra={
                    "event": "analysis_failed",
                    "stage": current_stage,
                    "provider": self.provider_name,
                    "model": self.model,
                    "total_time_ms": total_time,
                },
            )

            raise