"""
Orquestrador principal do AI Business Analyst.

Responsabilidades:
- Identificar perguntas simples e compostas.
- Planejar até cinco objetivos independentes.
- Gerar, validar e executar SQL separadamente.
- Recuperar erros SQL com limite de correções.
- Permitir sucesso parcial entre consultas.
- Gerar interpretações e gráficos.
- Preservar compatibilidade com o fluxo anterior.
"""

import logging
import re
import unicodedata
from time import perf_counter

import duckdb

from ai_analyst.agent.cache import AnalysisCache
from ai_analyst.agent.capability_validator import (
    UnsupportedAnalysisError,
    validate_analysis_capability,
)
from ai_analyst.agent.query_planner import QueryPlanner
from ai_analyst.agent.response_generator import ResponseGenerator
from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.agent.sql_generator import SQLGenerator
from ai_analyst.data.db import get_database_version
from ai_analyst.llm.factory import create_llm_provider
from ai_analyst.security.guardrails import SQLSyntaxError

logger = logging.getLogger(__name__)


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

# Número máximo de correções semânticas por consulta.
MAX_SQL_CORRECTIONS = 2

# Limite absoluto de consultas independentes por análise.
MAX_SQL_QUERIES = 5


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
        self.query_planner = None

        # Etapas da análise atual.
        self.steps: list[dict] = []

        # Preserva o último SQL gerado.
        self.last_sql: str | None = None

    def _initialize_ai_components(self):
        """Inicializa os componentes usando o provedor selecionado."""

        if self.provider is None:
            if self.gemini is not None:
                self.provider = self.gemini

            elif self.provider_name == "gemini":
                self.provider = create_llm_provider()

            else:
                self.provider = create_llm_provider(self.provider_name)

        self.model = getattr(
            self.provider,
            "model",
            DEFAULT_MODELS[self.provider_name],
        )

        self.gemini = self.provider

        if self.sql_generator is None:
            self.sql_generator = SQLGenerator(provider=self.provider)

        if self.response_generator is None:
            self.response_generator = ResponseGenerator(provider=self.provider)

        if self.query_planner is None:
            self.query_planner = QueryPlanner(provider=self.provider)

    # ==========================================================
    # CLASSIFICAÇÃO DE PERGUNTAS
    # ==========================================================

    @staticmethod
    def _is_numeric(value) -> bool:
        """Verifica se um valor é numérico, excluindo booleanos."""

        return isinstance(value, (int, float)) and not isinstance(value, bool)

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

    @classmethod
    def _requires_query_planning(cls, question: str) -> bool:
        """
        Identifica perguntas que provavelmente contêm
        objetivos de negócio independentes.

        A classificação é local e não utiliza uma chamada adicional
        ao LLM para perguntas aparentemente simples.
        """

        normalized = cls._normalize_text(question)

        strong_indicators = (
            r"\balem disso\b",
            r"\bseparadamente\b",
            r"\bindependentemente\b",
            r"\be tambem\b",
            r"\btambem quero saber\b",
            r"\btambem analise\b",
            r"\be ainda\b",
            r"\balem de\b",
        )

        if ";" in normalized:
            return True

        if any(re.search(pattern, normalized) for pattern in strong_indicators):
            return True

        if not re.search(r"\b(e|tambem)\b", normalized):
            return False

        metric_groups = {
            "orders": (
                r"\b(pedidos?|ordens?|volume de pedidos)\b",
                r"\bquantidade de pedidos\b",
            ),
            "revenue": (r"\b(faturamento|receita|vendas|valor vendido)\b",),
            "average": (r"\b(ticket medio|media|valor medio|average)\b",),
            "cancellation": (r"\b(cancelamento|cancelamentos|taxa de cancelamento)\b",),
            "delivery": (r"\b(tempo de entrega|tempo de preparo|atraso|atrasos)\b",),
            "rating": (r"\b(avaliacao|avaliacoes|nota|notas|satisfacao)\b",),
            "customers": (r"\b(clientes|consumidores)\b",),
            "restaurants": (r"\b(restaurantes)\b",),
            "couriers": (r"\b(entregadores|entregador)\b",),
        }

        detected_groups = set()

        for group_name, patterns in metric_groups.items():
            if any(re.search(pattern, normalized) for pattern in patterns):
                detected_groups.add(group_name)

        return len(detected_groups) >= 2

    # ==========================================================
    # INTELIGÊNCIA DE VISUALIZAÇÃO
    # ==========================================================

    @classmethod
    def _is_time_column(cls, column: str) -> bool:
        """
        Identifica se uma coluna representa uma dimensão temporal.

        Utiliza palavras completas, evitando classificações
        indevidas causadas por simples correspondências parciais.

        Exemplos reconhecidos:
        - order_date
        - order_month
        - delivery_year
        - created_at
        - month
        - year
        - semana
        - trimestre
        """

        time_terms = {
            "date",
            "datetime",
            "timestamp",
            "month",
            "year",
            "day",
            "week",
            "quarter",
            "period",
            "created",
            "updated",
            "data",
            "mes",
            "ano",
            "dia",
            "semana",
            "trimestre",
            "periodo",
        }

        column_tokens = cls._tokenize(column)

        return bool(column_tokens.intersection(time_terms))

    @classmethod
    def _is_ranking_question(cls, question: str) -> bool:
        """
        Identifica perguntas de ranking.

        Rankings e listas de maiores ou menores valores devem
        priorizar gráficos de barras, não gráficos de pizza.
        """

        ranking_terms = {
            "top",
            "ranking",
            "rank",
            "maiores",
            "maior",
            "menores",
            "menor",
            "melhores",
            "melhor",
            "piores",
            "pior",
            "primeiros",
            "primeiras",
            "ultimos",
            "ultimas",
            "mais",
            "menos",
        }

        question_tokens = cls._tokenize(question)

        return bool(question_tokens.intersection(ranking_terms))

    @classmethod
    def _is_comparison_question(cls, question: str) -> bool:
        """
        Identifica perguntas que solicitam comparações explícitas.
        """

        comparison_terms = {
            "comparar",
            "compare",
            "comparacao",
            "comparativo",
            "comparativos",
            "versus",
            "vs",
            "confrontar",
            "diferenca",
            "diferencas",
        }

        question_tokens = cls._tokenize(question)

        return bool(question_tokens.intersection(comparison_terms))

    @classmethod
    def _is_distribution_question(cls, question: str) -> bool:
        """
        Identifica perguntas que solicitam proporções,
        participação ou distribuição de valores.
        """

        distribution_terms = {
            "distribuicao",
            "distribuir",
            "proporcao",
            "proporcoes",
            "participacao",
            "percentual",
            "porcentagem",
            "composicao",
            "share",
            "breakdown",
            "percentage",
            "proportion",
        }

        question_tokens = cls._tokenize(question)

        return bool(question_tokens.intersection(distribution_terms))

    # ==========================================================
    # CONSTRUÇÃO DE GRÁFICOS
    # ==========================================================

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
            raise ValueError("Nenhuma coluna numérica disponível para o gráfico.")

        question_tokens = cls._tokenize(question)

        metric_groups = {
            "cancellation_rate": {
                "question": {
                    "cancelamento",
                    "cancelamentos",
                    "cancelado",
                    "cancelados",
                    "cancelada",
                    "canceladas",
                    "taxa",
                    "percentual",
                    "porcentagem",
                    "percent",
                    "percentage",
                    "rate",
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
                    "cancellation",
                    "cancelation",
                    "cancelamento",
                    "cancelados",
                    "canceladas",
                    "rate",
                    "percentage",
                    "percentual",
                    "taxa",
                },
            },
            "revenue": {
                "question": {
                    "faturamento",
                    "receita",
                    "vendas",
                    "venda",
                    "financeiro",
                    "arrecadacao",
                    "monetario",
                },
                "columns": {
                    "revenue",
                    "total_revenue",
                    "total_sales",
                    "sales",
                    "faturamento",
                    "receita",
                    "total_revenue_amount",
                },
                "column_tokens": {
                    "revenue",
                    "sales",
                    "sale",
                    "faturamento",
                    "receita",
                    "amount",
                    "valor",
                    "total",
                },
            },
            "orders": {
                "question": {
                    "pedido",
                    "pedidos",
                    "ordem",
                    "ordens",
                    "volume",
                    "quantidade",
                    "total",
                },
                "columns": {
                    "total_orders",
                    "orders_count",
                    "order_count",
                    "orders",
                    "pedidos",
                    "quantidade_pedidos",
                },
                "column_tokens": {
                    "orders",
                    "order",
                    "pedidos",
                    "pedido",
                    "count",
                    "quantidade",
                    "total",
                },
            },
            "canceled_orders": {
                "question": {
                    "cancelado",
                    "cancelados",
                    "cancelada",
                    "canceladas",
                    "cancelamento",
                    "cancelamentos",
                },
                "columns": {
                    "canceled_orders",
                    "cancelled_orders",
                    "cancelados",
                    "pedidos_cancelados",
                    "total_canceled_orders",
                },
                "column_tokens": {
                    "canceled",
                    "cancelled",
                    "cancelados",
                    "canceladas",
                    "cancelamento",
                },
            },
            "delivered_orders": {
                "question": {
                    "entregue",
                    "entregues",
                    "entregados",
                    "entregadas",
                    "entrega",
                    "entregas",
                },
                "columns": {
                    "delivered_orders",
                    "orders_delivered",
                    "pedidos_entregues",
                    "total_delivered_orders",
                },
                "column_tokens": {
                    "delivered",
                    "entregues",
                    "entregue",
                    "entregados",
                    "entregadas",
                },
            },
            "average": {
                "question": {
                    "media",
                    "medio",
                    "ticket",
                    "average",
                    "avg",
                    "promedio",
                },
                "columns": {
                    "average",
                    "avg",
                    "average_order_value",
                    "average_ticket",
                    "ticket_medio",
                    "media",
                },
                "column_tokens": {
                    "average",
                    "avg",
                    "media",
                    "medio",
                    "ticket",
                },
            },
            "percentage": {
                "question": {
                    "percentual",
                    "porcentagem",
                    "percent",
                    "percentage",
                    "participacao",
                    "proporcao",
                    "taxa",
                    "share",
                },
                "columns": {
                    "percentage",
                    "percent",
                    "rate",
                    "percentage_rate",
                    "share",
                    "proportion",
                    "percentual",
                    "taxa",
                },
                "column_tokens": {
                    "percentage",
                    "percent",
                    "rate",
                    "share",
                    "proportion",
                    "percentual",
                    "taxa",
                },
            },
        }

        requested_groups = []

        for group_name, config in metric_groups.items():
            if question_tokens.intersection(config["question"]):
                requested_groups.append(group_name)

        if "cancellation_rate" in requested_groups or (
            "canceled_orders" in requested_groups and "percentage" in requested_groups
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

                matching_tokens = column_tokens.intersection(config["column_tokens"])

                score += len(matching_tokens) * 10

                if group_name == "cancellation_rate":
                    if "rate" in column_tokens or "taxa" in column_tokens:
                        score += 30

                    if "percentage" in column_tokens or "percentual" in column_tokens:
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
            if not any(label_key.startswith(prefix) for prefix in ranking_prefixes):
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
                    if key in columns and self._is_numeric(result.get(key))
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
                    if key in columns and self._is_numeric(result.get(key))
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
                "title": (f"{self._humanize_column_name(value_name)} por categoria"),
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
        """
        Seleciona a visualização mais adequada com base
        na pergunta e na estrutura dos resultados.

        Estratégia:
        - Séries temporais: linha.
        - Métricas temporais independentes: barras agrupadas.
        - Comparações temporais por categoria: barras agrupadas.
        - Rankings: barras.
        - Comparações: barras.
        - Distribuições categóricas pequenas: pizza.
        - Categorias e métricas gerais: barras.
        """

        if not results:
            return None

        if len(results) == 1:
            comparison_chart = self._build_single_row_comparison(results[0])

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
                row.get(column) is not None and self._is_numeric(row.get(column))
                for row in results
            )
        ]

        categorical_columns = [
            column
            for column in columns
            if column not in numeric_columns
            and all(row.get(column) is not None for row in results)
        ]

        if not numeric_columns or not categorical_columns:
            return None

        proportion_terms = {
            "status",
            "category",
            "categoria",
            "type",
            "tipo",
        }

        series_terms = {
            "status",
            "category",
            "categoria",
            "type",
            "tipo",
            "method",
            "payment",
        }

        # ------------------------------------------------------
        # DIMENSÕES TEMPORAIS
        # ------------------------------------------------------

        time_columns = [
            column for column in categorical_columns if self._is_time_column(column)
        ]

        # Prioriza a dimensão temporal no eixo horizontal.
        label_candidates = time_columns or categorical_columns

        label_key = max(
            label_candidates,
            key=lambda column: len({str(row[column]) for row in results}),
        )

        categories = {str(row[label_key]) for row in results}

        # Evita gráficos com excesso de categorias.
        if len(categories) > 20:
            return None

        # ------------------------------------------------------
        # FORMATO LARGO: MÚLTIPLAS COLUNAS NUMÉRICAS
        # ------------------------------------------------------
        # Cada coluna numérica representa uma série diferente.
        #
        # Exemplo:
        #
        # {
        #     "month": "2025-01",
        #     "delivered_orders": 4901,
        #     "canceled_orders": 186
        # }
        #
        # Resultado:
        #
        # {
        #     "month": "2025-01",
        #     "series_0": 4901,
        #     "series_1": 186
        # }
        #
        # A propriedade "series" fornece os nomes que serão
        # utilizados pelo frontend na legenda do gráfico.
        # ------------------------------------------------------

        if time_columns and 2 <= len(numeric_columns) <= 8:
            wide_series_columns = numeric_columns

            series_metadata = [
                {
                    "data_key": f"series_{index}",
                    "label": self._humanize_column_name(column),
                }
                for index, column in enumerate(wide_series_columns)
            ]

            wide_chart_data = []
            seen_categories = set()
            has_duplicate_category = False

            for row in results:
                category_value = str(row[label_key])

                # Evita sobrescrever silenciosamente resultados
                # de meses ou períodos repetidos.
                if category_value in seen_categories:
                    has_duplicate_category = True
                    break

                seen_categories.add(category_value)

                chart_row = {
                    label_key: category_value,
                }

                for index, column in enumerate(wide_series_columns):
                    chart_row[f"series_{index}"] = float(row[column])

                wide_chart_data.append(chart_row)

            if not has_duplicate_category:
                series_labels = [item["label"] for item in series_metadata]

                series_title = " e ".join(series_labels)

                logger.info(
                    "Gráfico temporal com múltiplas métricas numéricas gerado",
                    extra={
                        "event": "wide_grouped_chart_generated",
                        "label_key": label_key,
                        "numeric_columns": wide_series_columns,
                        "series_count": len(series_metadata),
                        "category_count": len(wide_chart_data),
                    },
                )

                return {
                    "type": "bar",
                    "title": (
                        f"{series_title} por {self._humanize_column_name(label_key)}"
                    ),
                    "label_key": label_key,
                    "value_key": wide_series_columns[0],
                    "series": series_metadata,
                    "data": wide_chart_data,
                }

            logger.warning(
                "Gráfico temporal com múltiplas métricas ignorado "
                "por categorias duplicadas",
                extra={
                    "event": "wide_grouped_chart_duplicate_category",
                    "label_key": label_key,
                    "numeric_columns": wide_series_columns,
                },
            )

        # ------------------------------------------------------
        # SELEÇÃO DA MÉTRICA PRINCIPAL
        # ------------------------------------------------------

        value_key = self._select_value_column(
            question=question,
            numeric_columns=numeric_columns,
        )

        # ------------------------------------------------------
        # IDENTIFICAÇÃO DA SEGUNDA DIMENSÃO CATEGÓRICA
        # ------------------------------------------------------
        # Exemplos:
        # - Mês e status.
        # - Mês e categoria.
        # - Mês e método de pagamento.
        # ------------------------------------------------------

        series_candidates = [
            column
            for column in categorical_columns
            if column != label_key
            and 2 <= len({str(row[column]) for row in results}) <= 8
        ]

        series_key = None

        if time_columns and series_candidates:
            series_key = max(
                series_candidates,
                key=lambda column: (
                    bool(self._tokenize(column).intersection(series_terms)),
                    len({str(row[column]) for row in results}),
                ),
            )

        # ------------------------------------------------------
        # FORMATO LONGO: MÚLTIPLAS SÉRIES CATEGÓRICAS
        # ------------------------------------------------------
        # Cada valor da dimensão secundária recebe uma série
        # própria, identificada por series_0, series_1 etc.
        #
        # Exemplo:
        # {
        #     "order_month": "2025-01",
        #     "status": "Delivered",
        #     "total_orders": 4901
        # }
        #
        # A propriedade "series" informa ao frontend o nome
        # legível e a chave de cada série.
        # ------------------------------------------------------

        if series_key is not None:
            series_values = sorted({str(row[series_key]) for row in results})

            if len(series_values) >= 2:
                series_data_keys = {}
                chart_series = []

                for index, series_value in enumerate(series_values):
                    data_key = f"series_{index}"

                    series_data_keys[series_value] = data_key

                    chart_series.append(
                        {
                            "data_key": data_key,
                            "label": series_value.strip().title(),
                        }
                    )

                grouped_rows = {}
                has_duplicate_combination = False

                for row in results:
                    label_value = str(row[label_key])
                    series_value = str(row[series_key])

                    data_key = series_data_keys[series_value]

                    chart_row = grouped_rows.setdefault(
                        label_value,
                        {
                            label_key: label_value,
                        },
                    )

                    # Evita agregar silenciosamente registros repetidos.
                    if data_key in chart_row:
                        has_duplicate_combination = True
                        break

                    chart_row[data_key] = float(row[value_key])

                if not has_duplicate_combination:
                    # Mantém a ordem original das categorias.
                    grouped_data = list(grouped_rows.values())

                    logger.info(
                        "Gráfico com múltiplas séries gerado",
                        extra={
                            "event": "grouped_chart_generated",
                            "label_key": label_key,
                            "series_key": series_key,
                            "value_key": value_key,
                            "series_count": len(series_values),
                            "category_count": len(grouped_rows),
                        },
                    )

                    return {
                        "type": "bar",
                        "title": (
                            f"{self._humanize_column_name(value_key)} "
                            f"por {self._humanize_column_name(label_key)} "
                            f"e {self._humanize_column_name(series_key)}"
                        ),
                        "label_key": label_key,
                        "value_key": value_key,
                        "series": chart_series,
                        "data": grouped_data,
                    }

                logger.warning(
                    "Gráfico agrupado ignorado por combinações duplicadas",
                    extra={
                        "event": "grouped_chart_duplicate_combination",
                        "label_key": label_key,
                        "series_key": series_key,
                    },
                )

        # ------------------------------------------------------
        # GRÁFICOS DE UMA ÚNICA SÉRIE
        # ------------------------------------------------------

        label_tokens = self._tokenize(label_key)

        is_temporal_chart = self._is_time_column(label_key)

        is_ranking = self._is_ranking_question(question)

        is_comparison = self._is_comparison_question(question)

        is_distribution = self._is_distribution_question(question)

        is_proportion_dimension = bool(label_tokens.intersection(proportion_terms))

        # ------------------------------------------------------
        # DECISÃO DO TIPO DE GRÁFICO
        # ------------------------------------------------------
        #
        # 1. Linha:
        #    Evolução de uma métrica ao longo do tempo.
        #
        # 2. Barras:
        #    Rankings, comparações e categorias gerais.
        #
        # 3. Pizza:
        #    Distribuição e proporção de poucas categorias.
        #
        # A dimensão temporal tem prioridade sobre palavras
        # de ranking, pois uma série histórica continua
        # representando evolução ao longo do tempo.
        # ------------------------------------------------------

        if is_temporal_chart:
            chart_type = "line"

        elif is_ranking or is_comparison:
            chart_type = "bar"

        elif len(categories) <= 8 and (is_distribution or is_proportion_dimension):
            chart_type = "pie"

        else:
            chart_type = "bar"

        chart_data = [
            {
                key: (
                    float(row[key]) if self._is_numeric(row.get(key)) else row.get(key)
                )
                for key in (label_key, value_key)
            }
            for row in results
        ]

        chart_config = {
            "type": chart_type,
            "title": (
                f"{self._humanize_column_name(value_key)} "
                f"por {self._humanize_column_name(label_key)}"
            ),
            "label_key": label_key,
            "value_key": value_key,
            "data": chart_data,
        }

        logger.info(
            "Configuração de gráfico gerada",
            extra={
                "event": "chart_configuration_generated",
                "chart_type": chart_type,
                "label_key": label_key,
                "value_key": value_key,
                "category_count": len(categories),
                "is_temporal": is_temporal_chart,
                "is_ranking": is_ranking,
                "is_comparison": is_comparison,
                "is_distribution": is_distribution,
            },
        )

        return chart_config

    # ==========================================================
    # EXECUÇÃO DE UMA CONSULTA PLANEJADA
    # ==========================================================

    def _execute_planned_query(
        self,
        planned_query: dict,
        record_stage,
        run_stage,
    ) -> dict:
        """
        Gera e executa uma consulta independente.

        Cada consulta possui seu próprio limite de correções.
        Erros SQL são registrados individualmente e não impedem
        a execução das próximas consultas.
        """

        query_id = planned_query["id"]
        objective = planned_query["objective"]
        query_question = planned_query["question"]

        prefix = f"query_{query_id}"

        # Verifica se o banco possui a granularidade necessária
        # antes de consumir uma chamada do provedor de LLM.
        capability_start = perf_counter()

        try:
            run_stage(
                f"{prefix}_capability_validation",
                lambda: validate_analysis_capability(
                    objective=objective,
                    question=query_question,
                ),
            )

        except UnsupportedAnalysisError as exc:
            capability_validation_ms = round(
                (perf_counter() - capability_start) * 1000,
                2,
            )

            logger.warning(
                "Objetivo não suportado pelo schema",
                extra={
                    "event": "planned_query_unsupported",
                    "query_id": query_id,
                    "objective": objective,
                    "failure_type": "unsupported_capability",
                },
            )

            return {
                "id": query_id,
                "objective": objective,
                "question": query_question,
                "status": "failure",
                "failure_type": "unsupported_capability",
                "sql": None,
                "results": [],
                "total_results": 0,
                "chart": None,
                "error": str(exc),
                "metrics": {
                    "capability_validation_ms": capability_validation_ms,
                    "sql_generation_ms": 0.0,
                    "sql_correction_ms": 0.0,
                    "sql_execution_ms": 0.0,
                    "total_time_ms": capability_validation_ms,
                },
            }

        capability_validation_ms = round(
            (perf_counter() - capability_start) * 1000,
            2,
        )

        sql = None
        pending_error = None

        generation_start = perf_counter()

        try:
            try:
                sql = run_stage(
                    f"{prefix}_sql_generation",
                    lambda: self.sql_generator.generate(query_question),
                )

                self.last_sql = sql

            except SQLSyntaxError as exc:
                pending_error = exc
                sql = getattr(exc, "sql", None)
                self.last_sql = sql

            sql_generation_ms = round(
                (perf_counter() - generation_start) * 1000,
                2,
            )

            correction_count = 0
            correction_ms = 0.0
            execution_elapsed = 0.0
            results = None

            def query_stage_callback(
                name,
                status,
                duration_ms,
                details=None,
            ):
                record_stage(
                    f"{prefix}_{name}",
                    status,
                    duration_ms,
                    details,
                )

            self.sql_executor.set_stage_callback(query_stage_callback)

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

                        correction_start = perf_counter()

                        try:
                            sql = run_stage(
                                (f"{prefix}_sql_correction_{correction_number}"),
                                lambda failed_sql=failed_sql, error_message=error_message: (
                                    self.sql_generator.correct_sql(
                                        question=query_question,
                                        failed_sql=failed_sql,
                                        error_message=error_message,
                                    )
                                ),
                            )

                            self.last_sql = sql
                            pending_error = None

                        except SQLSyntaxError as exc:
                            pending_error = exc
                            sql = getattr(exc, "sql", None)
                            self.last_sql = sql

                        finally:
                            correction_ms += (perf_counter() - correction_start) * 1000

                        continue

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

                        self.last_sql = getattr(exc, "sql", None) or sql

                        sql = self.last_sql

                    finally:
                        execution_elapsed += perf_counter() - execution_start

            finally:
                self.sql_executor.set_stage_callback(None)

            sql_execution_ms = round(
                execution_elapsed * 1000,
                2,
            )

            total_query_time = round(
                capability_validation_ms
                + sql_generation_ms
                + correction_ms
                + sql_execution_ms,
                2,
            )

            return {
                "id": query_id,
                "objective": objective,
                "question": query_question,
                "status": "success",
                "sql": sql,
                "results": results,
                "total_results": len(results),
                "chart": None,
                "error": None,
                "metrics": {
                    "capability_validation_ms": capability_validation_ms,
                    "sql_generation_ms": sql_generation_ms,
                    "sql_correction_ms": round(correction_ms, 2),
                    "sql_execution_ms": sql_execution_ms,
                    "total_time_ms": total_query_time,
                },
            }

        except (
            SQLSyntaxError,
            duckdb.ProgrammingError,
        ) as exc:
            logger.warning(
                "Consulta planejada falhou",
                extra={
                    "event": "planned_query_failed",
                    "query_id": query_id,
                    "objective": objective,
                    "error_type": type(exc).__name__,
                },
            )

            return {
                "id": query_id,
                "objective": objective,
                "question": query_question,
                "status": "failure",
                "sql": sql,
                "results": [],
                "total_results": 0,
                "chart": None,
                "error": str(exc),
                "metrics": {
                    "sql_generation_ms": round(
                        (perf_counter() - generation_start) * 1000,
                        2,
                    ),
                },
            }

    # ==========================================================
    # ANÁLISE COM MÚLTIPLAS CONSULTAS
    # ==========================================================

    def _analyze_multi(
        self,
        question: str,
        data_version,
        analysis_start: float,
        record_stage,
        run_stage,
    ) -> dict:
        """Executa uma análise composta com até cinco consultas."""

        planning_start = perf_counter()

        plan = run_stage(
            "query_planning",
            lambda: self.query_planner.plan(question),
        )

        planning_ms = round(
            (perf_counter() - planning_start) * 1000,
            2,
        )

        if not isinstance(plan, list) or not plan:
            raise ValueError("O planejador não retornou objetivos válidos.")

        if len(plan) > MAX_SQL_QUERIES:
            raise ValueError(
                f"O planejador excedeu o limite de {MAX_SQL_QUERIES} consultas."
            )

        queries = []

        for planned_query in plan:
            query_result = self._execute_planned_query(
                planned_query=planned_query,
                record_stage=record_stage,
                run_stage=run_stage,
            )

            if query_result["status"] == "success":
                query_result["chart"] = run_stage(
                    f"query_{query_result['id']}_chart_generation",
                    lambda query_result=query_result: self._build_chart_config(
                        query_result["results"],
                        question=query_result["question"],
                    ),
                )

            queries.append(query_result)

        # Preserva os resultados de cada consulta separadamente.
        successful_queries = [
            query for query in queries if query["status"] == "success"
        ]

        failed_queries = [
            query
            for query in queries
            if (
                query["status"] == "failure"
                and query.get("failure_type") != "unsupported_capability"
            )
        ]

        # A síntese utiliza os resultados individuais.
        response_generation_start = perf_counter()

        answer = run_stage(
            "multi_response_generation",
            lambda: self.response_generator.generate_multi(
                question=question,
                queries=queries,
            ),
        )

        response_generation_ms = round(
            (perf_counter() - response_generation_start) * 1000,
            2,
        )

        # Expõe limitações do schema de forma determinística.
        # Não depende exclusivamente da interpretação do LLM.
        unsupported_queries = [
            query
            for query in queries
            if query.get("failure_type") == "unsupported_capability"
        ]

        if unsupported_queries:
            limitation_lines = [
                f"- {query['objective']}: {query['error']}"
                for query in unsupported_queries
            ]

            answer = f"{answer.rstrip()}\n\nLimitações dos dados:\n" + "\n".join(
                limitation_lines
            )

        # Compatibilidade com o contrato antigo:
        # os campos singulares representam a primeira consulta bem-sucedida.
        primary_query = successful_queries[0] if successful_queries else queries[0]

        sql_generation_ms = round(
            sum(
                query.get("metrics", {}).get(
                    "sql_generation_ms",
                    0,
                )
                for query in queries
            ),
            2,
        )

        sql_execution_ms = round(
            sum(
                query.get("metrics", {}).get(
                    "sql_execution_ms",
                    0,
                )
                for query in queries
            ),
            2,
        )

        total_time = round(
            (perf_counter() - analysis_start) * 1000,
            2,
        )

        analysis_result = {
            "question": question,
            # Contrato legado: consulta principal.
            "sql": primary_query.get("sql"),
            "results": primary_query.get("results", []),
            "total_results": primary_query.get(
                "total_results",
                0,
            ),
            "answer": answer,
            "chart": primary_query.get("chart"),
            # Novo contrato: resultados independentes.
            "queries": queries,
            "query_count": len(queries),
            "successful_queries": len(successful_queries),
            "failed_queries": len(failed_queries),
            "unsupported_queries": len(unsupported_queries),
            "partial_success": (
                len(successful_queries) > 0
                and (len(failed_queries) > 0 or len(unsupported_queries) > 0)
            ),
            "analysis_mode": "multi_query",
            "provider": self.provider_name,
            "model": self.model,
            "cache_hit": False,
            "steps": self.steps,
            "metrics": {
                "total_time_ms": total_time,
                "query_planning_ms": planning_ms,
                "sql_generation_ms": sql_generation_ms,
                "sql_execution_ms": sql_execution_ms,
                "response_generation_ms": response_generation_ms,
                "query_count": len(queries),
                "successful_queries": len(successful_queries),
                "failed_queries": len(failed_queries),
                "unsupported_queries": len(unsupported_queries),
                "cache_hit": False,
            },
        }

        self.cache.set(
            question=question,
            data_version=data_version,
            result=analysis_result,
            provider=self.provider_name,
            model=self.model,
        )

        logger.info(
            "Análise composta concluída",
            extra={
                "event": "multi_analysis_completed",
                "provider": self.provider_name,
                "model": self.model,
                "query_count": len(queries),
                "successful_queries": len(successful_queries),
                "failed_queries": len(failed_queries),
                "unsupported_queries": len(unsupported_queries),
                "total_time_ms": total_time,
            },
        )

        return analysis_result

    # ==========================================================
    # ORQUESTRADOR PRINCIPAL
    # ==========================================================

    def analyze(self, question: str) -> dict:
        """Gera SQL, executa consultas e interpreta os resultados."""

        self.steps = []
        self.last_sql = None

        if not isinstance(question, str) or not question.strip():
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

            except UnsupportedAnalysisError as exc:
                duration_ms = round(
                    (perf_counter() - stage_start) * 1000,
                    2,
                )

                record_stage(
                    name,
                    "unsupported",
                    duration_ms,
                    str(exc),
                )
                raise

            except Exception:
                duration_ms = round(
                    (perf_counter() - stage_start) * 1000,
                    2,
                )

                record_stage(name, "failed", duration_ms)
                raise

            duration_ms = round(
                (perf_counter() - stage_start) * 1000,
                2,
            )

            record_stage(name, "completed", duration_ms)

            return result

        try:
            # 1. Identificar versão do banco.
            data_version = run_stage(
                "database_version",
                get_database_version,
            )

            # 2. Consultar cache.
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
                # Recupera gráficos de análises compostas antigas.
                cached_queries = cached_result.get("queries", [])

                if cached_queries:
                    for query in cached_queries:
                        if (
                            query.get("status") == "success"
                            and query.get("chart") is None
                            and query.get("results")
                        ):
                            query["chart"] = run_stage(
                                (f"query_{query['id']}_chart_generation"),
                                lambda query=query: self._build_chart_config(
                                    query["results"],
                                    question=query["question"],
                                ),
                            )

                    primary_query = next(
                        (
                            query
                            for query in cached_queries
                            if query.get("status") == "success"
                        ),
                        cached_queries[0],
                    )

                    cached_result["chart"] = primary_query.get("chart")

                elif cached_result.get("chart") is None and cached_result.get(
                    "results"
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
                    (perf_counter() - analysis_start) * 1000,
                    2,
                )

                record_stage(
                    "cache_hit",
                    "completed",
                    0,
                    "Análise recuperada do cache. Nenhuma chamada à IA foi realizada.",
                )

                cached_result["steps"] = steps

                cached_result.setdefault(
                    "metrics",
                    {},
                )

                cached_result["metrics"]["total_time_ms"] = total_time
                cached_result["metrics"]["cache_hit"] = True

                return cached_result

            logger.info(
                "Cache miss: iniciando nova análise",
                extra={
                    "event": "analysis_cache_miss",
                    "provider": self.provider_name,
                    "model": self.model,
                },
            )

            # 3. Inicializar componentes.
            run_stage(
                "ai_initialization",
                self._initialize_ai_components,
            )

            # 4. Identificar se a pergunta é composta.
            if self._requires_query_planning(question):
                return self._analyze_multi(
                    question=question,
                    data_version=data_version,
                    analysis_start=analysis_start,
                    record_stage=record_stage,
                    run_stage=run_stage,
                )

            # ==================================================
            # FLUXO LEGADO: UMA ÚNICA CONSULTA
            # ==================================================

            # 5. Gerar SQL.
            sql_generation_start = perf_counter()
            pending_error = None
            sql = None

            try:
                sql = run_stage(
                    "sql_generation",
                    lambda: self.sql_generator.generate(question),
                )

                self.last_sql = sql

            except SQLSyntaxError as exc:
                pending_error = exc
                sql = exc.sql
                self.last_sql = sql

            sql_generation_ms = round(
                (perf_counter() - sql_generation_start) * 1000,
                2,
            )

            # 6. Executar SQL com correções limitadas.
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
                                lambda failed_sql=failed_sql, error_message=error_message: (
                                    self.sql_generator.correct_sql(
                                        question=question,
                                        failed_sql=failed_sql,
                                        error_message=error_message,
                                    )
                                ),
                            )

                            self.last_sql = sql
                            pending_error = None

                        except SQLSyntaxError as exc:
                            pending_error = exc
                            sql = exc.sql
                            self.last_sql = sql

                            continue

                    current_stage = f"sql_execution_attempt_{correction_count + 1}"

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

                        self.last_sql = getattr(exc, "sql", None) or sql

                        sql = self.last_sql

                    finally:
                        sql_execution_elapsed += perf_counter() - execution_start

            finally:
                self.sql_executor.set_stage_callback(None)

            sql_execution_ms = round(
                sql_execution_elapsed * 1000,
                2,
            )

            # 7. Interpretar resultados.
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
                (perf_counter() - response_generation_start) * 1000,
                2,
            )

            # 8. Gerar gráfico.
            chart = run_stage(
                "chart_generation",
                lambda: self._build_chart_config(
                    results,
                    question=question,
                ),
            )

            # 9. Métricas.
            total_time = round(
                (perf_counter() - analysis_start) * 1000,
                2,
            )

            # 10. Montar resultado mantendo contrato legado.
            analysis_result = {
                "question": question,
                "sql": sql,
                "results": results,
                "total_results": len(results),
                "answer": answer,
                "chart": chart,
                "queries": [],
                "query_count": 1,
                "successful_queries": 1,
                "failed_queries": 0,
                "unsupported_queries": 0,
                "partial_success": False,
                "analysis_mode": "single_query",
                "provider": self.provider_name,
                "model": self.model,
                "cache_hit": False,
                "steps": steps,
                "metrics": {
                    "total_time_ms": total_time,
                    "sql_generation_ms": sql_generation_ms,
                    "sql_execution_ms": sql_execution_ms,
                    "response_generation_ms": response_generation_ms,
                    "query_count": 1,
                    "successful_queries": 1,
                    "failed_queries": 0,
                    "cache_hit": False,
                },
            }

            # 11. Armazenar no cache.
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
                (perf_counter() - analysis_start) * 1000,
                2,
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
