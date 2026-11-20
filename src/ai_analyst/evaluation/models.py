"""Modelos de dados do AI Evaluation Engine.

Define as estruturas utilizadas para representar:
- Cenários de avaliação.
- Resultados individuais.
- Relatórios consolidados de execução.
- Estimativas de custos por cenário, modelo e provedor.

Implementado com dataclasses para evitar dependências externas.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Literal

ComparisonMode = Literal["exact", "tolerant"]

AnalysisMode = Literal[
    "single_query",
    "multi_query",
]

EvaluationType = Literal[
    "sql_result",
    "clarification",
    "unanswerable",
    "partial_answer",
]

ExpectedBehavior = Literal[
    "answer",
    "ask_clarification",
    "abstain",
    "partial_answer",
]

EvaluationStatus = Literal[
    "passed",
    "failed",
    "error",
    "skipped",
]


# ==========================================================
# COMPORTAMENTOS ESPERADOS
# ==========================================================


EXPECTED_BEHAVIOR_BY_TYPE: dict[EvaluationType, ExpectedBehavior] = {
    "sql_result": "answer",
    "clarification": "ask_clarification",
    "unanswerable": "abstain",
    "partial_answer": "partial_answer",
}


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================


def _non_negative_integer(value: Any) -> int:
    """Normaliza contadores inteiros não negativos."""

    if isinstance(value, bool) or not isinstance(value, int):
        return 0

    return max(value, 0)


def _non_negative_number(value: Any) -> float | None:
    """Normaliza valores numéricos financeiros não negativos."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return None

    if not isfinite(normalized) or normalized < 0:
        return None

    return normalized


def _normalize_string_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    """Normaliza uma coleção de textos para tupla validada.

    Strings isoladas e objetos de mapeamento não são aceitos
    como coleções, evitando conversões acidentais de caracteres
    ou chaves de dicionários.
    """

    if isinstance(value, (str, bytes, dict)):
        raise ValueError(
            f"O campo '{field_name}' deve ser uma coleção de textos, "
            "não uma string ou um mapeamento."
        )

    try:
        normalized = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"O campo '{field_name}' deve ser uma coleção de textos."
        ) from exc

    for item in normalized:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Todos os elementos de '{field_name}' devem ser textos não vazios."
            )

    if not allow_empty and not normalized:
        raise ValueError(f"O campo '{field_name}' não pode estar vazio.")

    return normalized


def _validate_optional_string(
    value: Any,
    field_name: str,
) -> None:
    """Valida um campo textual opcional."""

    if value is not None and not isinstance(value, str):
        raise TypeError(f"O campo '{field_name}' deve ser uma string ou None.")


def _validate_result_rows(
    rows: Any,
    field_name: str,
) -> None:
    """Valida a estrutura de uma coleção de resultados SQL."""

    if not isinstance(rows, list):
        raise TypeError(f"O campo '{field_name}' deve ser uma lista.")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"Cada registro de '{field_name}' deve ser um dicionário. "
                f"Registro inválido no índice {index}."
            )

        if any(not isinstance(column, str) for column in row):
            raise TypeError(
                f"Todas as colunas de '{field_name}' devem possuir nomes string. "
                f"Registro inválido no índice {index}."
            )


def _normalize_datetime(
    value: Any,
    field_name: str,
) -> datetime:
    """Valida uma data e normaliza datas sem fuso para UTC.

    Datas sem informação de fuso são interpretadas como UTC
    para evitar incompatibilidade entre timestamps ingênuos
    e timestamps com timezone.
    """

    if not isinstance(value, datetime):
        raise TypeError(f"O campo '{field_name}' deve ser um datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)

    return value


def _new_model_cost_bucket(
    provider: str,
    model: str,
    model_key: str,
) -> dict[str, Any]:
    """Cria um acumulador de custos para um modelo."""

    return {
        "provider": provider,
        "model": model,
        "model_key": model_key,
        "requests_total": 0,
        "requests_estimated": 0,
        "requests_unestimated": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "estimated_cost_usd": 0.0,
        "has_estimated_cost": False,
        "pricing_tiers": set(),
        "pricing_context_tiers": {},
        "pricing_contexts": {},
        "pricing_sources": set(),
        "pricing_as_of_dates": set(),
        "long_context_thresholds": set(),
    }


def _merge_model_costs(
    estimates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Consolida os custos por provedor e modelo."""

    aggregated: dict[str, dict[str, Any]] = {}

    for estimate in estimates:
        models = estimate.get("models", {})

        if not isinstance(models, dict):
            continue

        for model_key, model_data in models.items():
            if not isinstance(model_key, str) or not model_key:
                continue

            if not isinstance(model_data, dict):
                continue

            provider = str(
                model_data.get("provider") or estimate.get("provider") or "unknown"
            )

            model = str(model_data.get("model") or model_key)

            qualified_key = f"{provider}/{model_key}"

            bucket = aggregated.setdefault(
                qualified_key,
                _new_model_cost_bucket(
                    provider=provider,
                    model=model,
                    model_key=model_key,
                ),
            )

            bucket["requests_total"] += _non_negative_integer(
                model_data.get("requests_total")
            )

            bucket["requests_estimated"] += _non_negative_integer(
                model_data.get("requests_estimated")
            )

            bucket["requests_unestimated"] += _non_negative_integer(
                model_data.get("requests_unestimated")
            )

            bucket["input_tokens"] += _non_negative_integer(
                model_data.get("input_tokens")
            )

            bucket["output_tokens"] += _non_negative_integer(
                model_data.get("output_tokens")
            )

            bucket["thinking_tokens"] += _non_negative_integer(
                model_data.get("thinking_tokens")
            )

            model_cost = _non_negative_number(model_data.get("estimated_cost_usd"))

            if model_cost is not None:
                bucket["estimated_cost_usd"] += model_cost
                bucket["has_estimated_cost"] = True

            pricing_tier = model_data.get("pricing_tier")

            if isinstance(pricing_tier, str):
                bucket["pricing_tiers"].add(pricing_tier)

            pricing_source = model_data.get("pricing_source")

            if isinstance(pricing_source, str) and pricing_source:
                bucket["pricing_sources"].add(pricing_source)

            pricing_as_of = model_data.get("pricing_as_of")

            if isinstance(pricing_as_of, str) and pricing_as_of:
                bucket["pricing_as_of_dates"].add(pricing_as_of)

            threshold = model_data.get("long_context_threshold_tokens")

            if isinstance(threshold, int) and not isinstance(threshold, bool):
                bucket["long_context_thresholds"].add(threshold)

            context_tiers = model_data.get(
                "pricing_context_tiers",
                {},
            )

            if isinstance(context_tiers, dict):
                for context_name, count in context_tiers.items():
                    if not isinstance(context_name, str):
                        continue

                    bucket["pricing_context_tiers"][context_name] = bucket[
                        "pricing_context_tiers"
                    ].get(
                        context_name,
                        0,
                    ) + _non_negative_integer(count)

            pricing_contexts = model_data.get(
                "pricing_contexts",
                {},
            )

            if not isinstance(pricing_contexts, dict):
                continue

            for context_name, context_data in pricing_contexts.items():
                if not isinstance(context_name, str):
                    continue

                if not isinstance(context_data, dict):
                    continue

                context_bucket = bucket["pricing_contexts"].setdefault(
                    context_name,
                    {
                        "requests": 0,
                        "estimated_cost_usd": 0.0,
                        "has_estimated_cost": False,
                        "rates": None,
                        "rate_versions": [],
                        "rates_conflict": False,
                    },
                )

                context_bucket["requests"] += _non_negative_integer(
                    context_data.get("requests")
                )

                context_cost = _non_negative_number(
                    context_data.get("estimated_cost_usd")
                )

                if context_cost is not None:
                    context_bucket["estimated_cost_usd"] += context_cost
                    context_bucket["has_estimated_cost"] = True

                rates = context_data.get("rates")

                if isinstance(rates, dict):
                    rate_version = {
                        "pricing_as_of": pricing_as_of,
                        "rates": dict(rates),
                    }

                    if rate_version not in context_bucket["rate_versions"]:
                        context_bucket["rate_versions"].append(rate_version)

                    # Depois de detectar tarifas divergentes, não volta
                    # a apresentar uma tarifa única nas próximas iterações.
                    if not context_bucket["rates_conflict"]:
                        if context_bucket["rates"] is None:
                            context_bucket["rates"] = dict(rates)

                        elif context_bucket["rates"] != rates:
                            context_bucket["rates"] = None
                            context_bucket["rates_conflict"] = True

    result: dict[str, dict[str, Any]] = {}

    for key, bucket in sorted(aggregated.items()):
        pricing_sources = sorted(bucket["pricing_sources"])
        pricing_dates = sorted(bucket["pricing_as_of_dates"])
        pricing_tiers = sorted(bucket["pricing_tiers"])
        thresholds = sorted(bucket["long_context_thresholds"])

        contexts = {}

        for context_name, context_bucket in sorted(bucket["pricing_contexts"].items()):
            contexts[context_name] = {
                "requests": context_bucket["requests"],
                "estimated_cost_usd": (
                    round(
                        context_bucket["estimated_cost_usd"],
                        12,
                    )
                    if context_bucket["has_estimated_cost"]
                    else None
                ),
                "rates": context_bucket["rates"],
                "rate_versions": context_bucket["rate_versions"],
            }

        result[key] = {
            "provider": bucket["provider"],
            "model": bucket["model"],
            "requests_total": bucket["requests_total"],
            "requests_estimated": bucket["requests_estimated"],
            "requests_unestimated": bucket["requests_unestimated"],
            "input_tokens": bucket["input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "thinking_tokens": bucket["thinking_tokens"],
            "estimated_cost_usd": (
                round(
                    bucket["estimated_cost_usd"],
                    12,
                )
                if bucket["has_estimated_cost"]
                else None
            ),
            "pricing_tiers": pricing_tiers,
            "pricing_context_tiers": bucket["pricing_context_tiers"],
            "pricing_contexts": contexts,
            "pricing_sources": pricing_sources,
            "pricing_as_of_dates": pricing_dates,
            "long_context_threshold_tokens": (
                thresholds[0] if len(thresholds) == 1 else None
            ),
            "long_context_thresholds": thresholds,
        }

    return result


def _aggregate_provider_costs(
    models: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Consolida os custos por provedor."""

    providers: dict[str, dict[str, Any]] = {}

    for model_key, model_data in models.items():
        provider = model_data["provider"]

        bucket = providers.setdefault(
            provider,
            {
                "models": [],
                "requests_total": 0,
                "requests_estimated": 0,
                "requests_unestimated": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "estimated_cost_usd": 0.0,
                "has_estimated_cost": False,
            },
        )

        bucket["models"].append(model_key)

        for field_name in (
            "requests_total",
            "requests_estimated",
            "requests_unestimated",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
        ):
            bucket[field_name] += model_data[field_name]

        model_cost = model_data["estimated_cost_usd"]

        if model_cost is not None:
            bucket["estimated_cost_usd"] += model_cost
            bucket["has_estimated_cost"] = True

    result = {}

    for provider, bucket in sorted(providers.items()):
        result[provider] = {
            "models": sorted(bucket["models"]),
            "requests_total": bucket["requests_total"],
            "requests_estimated": bucket["requests_estimated"],
            "requests_unestimated": bucket["requests_unestimated"],
            "input_tokens": bucket["input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "thinking_tokens": bucket["thinking_tokens"],
            "estimated_cost_usd": (
                round(bucket["estimated_cost_usd"], 12)
                if bucket["has_estimated_cost"]
                else None
            ),
        }

    return result


def _aggregate_token_costs(
    results: list[EvaluationCaseResult],
) -> dict[str, Any]:
    """Consolida estimativas individuais em um resumo financeiro.

    A agregação financeira considera todos os cenários,
    independentemente do status da avaliação.

    Um cenário 'skipped' pode ter consumido tokens antes de
    ser classificado como não comparável. Portanto, seu custo
    não deve ser descartado.

    A ausência de estimativa em qualquer cenário torna o resumo
    incompleto, pois não é possível garantir que todo o consumo
    tenha sido contabilizado.
    """

    # ------------------------------------------------------
    # TODOS OS STATUS PARTICIPAM DA CONTABILIZAÇÃO FINANCEIRA.
    #
    # O status 'skipped' afeta a avaliação de qualidade,
    # mas não elimina o consumo de LLM que já aconteceu.
    # ------------------------------------------------------

    cost_accounted_results = list(results)

    estimates = [
        result.token_cost_estimate
        for result in cost_accounted_results
        if isinstance(result.token_cost_estimate, dict)
    ]

    missing_cost_data = len(cost_accounted_results) - len(estimates)

    requests_estimated = 0
    requests_unestimated = 0

    total_estimated_cost = 0.0
    has_estimated_cost = False

    currencies: set[str] = set()
    pricing_dates: set[str] = set()

    snapshots_complete = True

    for estimate in estimates:
        currency = estimate.get("currency")

        if isinstance(currency, str) and currency:
            currencies.add(currency)

        pricing_as_of = estimate.get("pricing_as_of")

        if isinstance(pricing_as_of, str) and pricing_as_of:
            pricing_dates.add(pricing_as_of)

        requests_estimated += _non_negative_integer(estimate.get("requests_estimated"))

        requests_unestimated += _non_negative_integer(
            estimate.get("requests_unestimated")
        )

        estimated_cost = _non_negative_number(estimate.get("estimated_cost_usd"))

        if estimated_cost is not None:
            total_estimated_cost += estimated_cost
            has_estimated_cost = True

        # Um snapshot só pode ser considerado completo
        # quando declara explicitamente is_complete=True.
        #
        # Ausência do campo, valor None ou valor inválido
        # não fornecem evidência suficiente de completude.
        if estimate.get("is_complete") is not True:
            snapshots_complete = False

    # As estimativas dos cenários skipped também chegam
    # normalmente às agregações por modelo e provedor.
    models = _merge_model_costs(estimates)

    providers = _aggregate_provider_costs(models)

    # ------------------------------------------------------
    # COMPLETUDE DO RESUMO
    # ------------------------------------------------------

    # Uma execução sem cenários não possui uma estimativa
    # financeira que possa ser considerada completa.
    #
    # Qualquer cenário sem snapshot, independentemente
    # do status, torna a contabilização incompleta.
    #
    # Solicitações sem estimativa e snapshots sem confirmação
    # explícita de completude também invalidam o resumo.
    is_complete = (
        bool(cost_accounted_results)
        and missing_cost_data == 0
        and requests_unestimated == 0
        and snapshots_complete
    )

    currency = (
        next(iter(currencies))
        if len(currencies) == 1
        else "mixed"
        if len(currencies) > 1
        else "USD"
    )

    pricing_as_of = next(iter(pricing_dates)) if len(pricing_dates) == 1 else None

    return {
        "currency": currency,
        "pricing_as_of": pricing_as_of,
        "pricing_as_of_dates": sorted(pricing_dates),
        "cases_with_cost_data": len(estimates),
        "cases_without_cost_data": missing_cost_data,
        "requests_estimated": requests_estimated,
        "requests_unestimated": requests_unestimated,
        "is_complete": is_complete,
        "estimated_cost_usd": (
            round(total_estimated_cost, 12) if has_estimated_cost else None
        ),
        "providers": providers,
        "models": models,
    }


# ==========================================================
# CENÁRIOS DE AVALIAÇÃO
# ==========================================================


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Representa um cenário individual de avaliação.

    Existem duas modalidades de avaliação:

    1. sql_result:
        Compara o resultado produzido pelo agente com uma consulta
        SQL de referência.

    2. Avaliação comportamental:
        Verifica se o agente apresenta o comportamento esperado
        diante de perguntas ambíguas, impossíveis de responder ou
        parcialmente respondíveis.

    Attributes:
        case_id: Identificador único do cenário.
        question: Pergunta enviada ao agente.
        reference_sql: Consulta SQL confiável usada como referência.
            Obrigatória para sql_result.
            Deve ser None para cenários comportamentais.
        category: Categoria de negócio do cenário.
        description: Explicação do objetivo da avaliação.
        comparison_mode: Política de comparação dos valores.
        absolute_tolerance: Tolerância numérica absoluta.
        relative_tolerance: Tolerância numérica relativa.
        required_columns: Colunas obrigatórias no resultado SQL.
        allow_extra_columns: Permite colunas adicionais.
        sort_rows: Ignora a ordem das linhas na comparação.
        tags: Etiquetas para agrupamento e filtragem.
        expected_analysis_mode: Modo de análise esperado pelo cenário.
            None permite qualquer modo.
            single_query exige uma única consulta.
            multi_query exige múltiplas consultas.
        evaluation_type: Modalidade de avaliação.
        expected_behavior: Comportamento esperado do agente.
            Quando None, é derivado automaticamente de evaluation_type.
        evaluation_criteria: Critérios utilizados para avaliar
            o comportamento do agente.
    """

    case_id: str
    question: str
    reference_sql: str | None
    category: str
    description: str = ""

    comparison_mode: ComparisonMode = "tolerant"

    absolute_tolerance: float = 0.01
    relative_tolerance: float = 0.0

    required_columns: tuple[str, ...] = ()

    allow_extra_columns: bool = False
    sort_rows: bool = True

    tags: tuple[str, ...] = ()

    expected_analysis_mode: AnalysisMode | None = None

    evaluation_type: EvaluationType = "sql_result"

    expected_behavior: ExpectedBehavior | None = None

    evaluation_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Valida os parâmetros do cenário."""

        # --------------------------------------------------
        # CAMPOS OBRIGATÓRIOS
        # --------------------------------------------------

        required_text_fields = {
            "case_id": self.case_id,
            "question": self.question,
            "category": self.category,
        }

        for field_name, value in required_text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"O campo '{field_name}' é obrigatório.")

        if not isinstance(self.description, str):
            raise TypeError("description deve ser uma string.")

        # --------------------------------------------------
        # MODALIDADE DE AVALIAÇÃO
        # --------------------------------------------------

        if not isinstance(self.evaluation_type, str):
            raise ValueError("evaluation_type deve ser uma string válida.")

        if self.evaluation_type not in EXPECTED_BEHAVIOR_BY_TYPE:
            raise ValueError(
                "evaluation_type deve ser um dos valores: "
                "'sql_result', 'clarification', "
                "'unanswerable' ou 'partial_answer'."
            )

        # --------------------------------------------------
        # COMPORTAMENTO ESPERADO
        # --------------------------------------------------

        expected_behavior = EXPECTED_BEHAVIOR_BY_TYPE[self.evaluation_type]

        if self.expected_behavior is None:
            object.__setattr__(
                self,
                "expected_behavior",
                expected_behavior,
            )

        elif (
            not isinstance(self.expected_behavior, str)
            or self.expected_behavior != expected_behavior
        ):
            raise ValueError(
                f"O comportamento esperado para '{self.evaluation_type}' "
                f"deve ser '{expected_behavior}'."
            )

        # --------------------------------------------------
        # SQL DE REFERÊNCIA
        # --------------------------------------------------

        if self.evaluation_type == "sql_result":
            if (
                not isinstance(self.reference_sql, str)
                or not self.reference_sql.strip()
            ):
                raise ValueError(
                    "O campo 'reference_sql' é obrigatório "
                    "para cenários do tipo 'sql_result'."
                )

        elif self.reference_sql is not None:
            raise ValueError(
                "Cenários comportamentais não devem possuir "
                "'reference_sql'. Utilize None."
            )

        # --------------------------------------------------
        # MODO DE COMPARAÇÃO
        # --------------------------------------------------

        if not isinstance(self.comparison_mode, str) or self.comparison_mode not in (
            "exact",
            "tolerant",
        ):
            raise ValueError("comparison_mode deve ser 'exact' ou 'tolerant'.")

        # --------------------------------------------------
        # MODO DE ANÁLISE
        # --------------------------------------------------

        if self.expected_analysis_mode is not None and (
            not isinstance(self.expected_analysis_mode, str)
            or self.expected_analysis_mode not in ("single_query", "multi_query")
        ):
            raise ValueError(
                "expected_analysis_mode deve ser None, 'single_query' ou 'multi_query'."
            )

        # --------------------------------------------------
        # FLAGS BOOLEANAS
        # --------------------------------------------------

        for field_name, value in (
            ("allow_extra_columns", self.allow_extra_columns),
            ("sort_rows", self.sort_rows),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} deve ser um booleano.")

        # --------------------------------------------------
        # TOLERÂNCIAS
        # --------------------------------------------------

        for field_name, value in (
            ("absolute_tolerance", self.absolute_tolerance),
            ("relative_tolerance", self.relative_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{field_name} deve ser um número finito não negativo."
                )

        # --------------------------------------------------
        # COLUNAS OBRIGATÓRIAS
        # --------------------------------------------------

        normalized_columns = _normalize_string_tuple(
            self.required_columns,
            "required_columns",
        )

        if len(set(normalized_columns)) != len(normalized_columns):
            raise ValueError("required_columns não pode conter nomes duplicados.")

        object.__setattr__(
            self,
            "required_columns",
            normalized_columns,
        )

        if self.evaluation_type != "sql_result" and normalized_columns:
            raise ValueError(
                "Cenários comportamentais não devem declarar required_columns."
            )

        # --------------------------------------------------
        # ETIQUETAS
        # --------------------------------------------------

        normalized_tags = _normalize_string_tuple(
            self.tags,
            "tags",
        )

        object.__setattr__(
            self,
            "tags",
            normalized_tags,
        )

        # --------------------------------------------------
        # CRITÉRIOS COMPORTAMENTAIS
        # --------------------------------------------------

        normalized_criteria = _normalize_string_tuple(
            self.evaluation_criteria,
            "evaluation_criteria",
        )

        if len(set(normalized_criteria)) != len(normalized_criteria):
            raise ValueError(
                "evaluation_criteria não pode conter critérios duplicados."
            )

        object.__setattr__(
            self,
            "evaluation_criteria",
            normalized_criteria,
        )

        if self.evaluation_type != "sql_result" and not normalized_criteria:
            raise ValueError(
                "Cenários comportamentais devem possuir "
                "pelo menos um evaluation_criteria."
            )

        # --------------------------------------------------
        # CONSISTÊNCIA DOS CENÁRIOS COMPORTAMENTAIS
        # --------------------------------------------------

        if self.evaluation_type != "sql_result":
            if self.expected_analysis_mode is not None:
                raise ValueError(
                    "Cenários comportamentais não devem definir expected_analysis_mode."
                )


# ==========================================================
# RESULTADOS INDIVIDUAIS
# ==========================================================


@dataclass(slots=True)
class EvaluationCaseResult:
    """Armazena o resultado da avaliação de um cenário.

    O status representa o resultado da execução:

    - passed: resultado aprovado.
    - failed: execução concluída, mas comparação reprovada.
    - error: erro durante a execução ou avaliação.
    - skipped: cenário não executado.

    token_cost_estimate armazena o snapshot de custos da análise
    associada ao cenário. É opcional para preservar compatibilidade.
    """

    case_id: str
    question: str
    status: EvaluationStatus

    score: float = 0.0

    expected_rows: list[dict[str, Any]] = field(default_factory=list)

    actual_rows: list[dict[str, Any]] = field(default_factory=list)

    differences: list[str] = field(default_factory=list)

    analysis_id: str | None = None

    provider: str | None = None
    model: str | None = None

    duration_ms: float | None = None

    cache_hit: bool | None = None

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    # Snapshot produzido por AnalysisLLMMetrics.snapshot().
    # Mantido no final para não deslocar argumentos posicionais.
    token_cost_estimate: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Valida a integridade do resultado individual."""

        # --------------------------------------------------
        # IDENTIFICAÇÃO
        # --------------------------------------------------

        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id deve ser uma string não vazia.")

        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question deve ser uma string não vazia.")

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        valid_statuses = {
            "passed",
            "failed",
            "error",
            "skipped",
        }

        if not isinstance(self.status, str) or self.status not in valid_statuses:
            raise ValueError(f"Status de avaliação inválido: {self.status!r}")

        # --------------------------------------------------
        # PONTUAÇÃO
        # --------------------------------------------------

        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
        ):
            raise ValueError("score deve ser um número finito entre 0.0 e 1.0.")

        # --------------------------------------------------
        # RESULTADOS SQL
        # --------------------------------------------------

        _validate_result_rows(self.expected_rows, "expected_rows")

        _validate_result_rows(self.actual_rows, "actual_rows")

        # --------------------------------------------------
        # DIFERENÇAS
        # --------------------------------------------------

        if not isinstance(self.differences, list):
            raise TypeError("differences deve ser uma lista de strings.")

        if any(not isinstance(item, str) for item in self.differences):
            raise TypeError("Todos os elementos de differences devem ser strings.")

        # --------------------------------------------------
        # CAMPOS TEXTUAIS OPCIONAIS
        # --------------------------------------------------

        for field_name, value in (
            ("analysis_id", self.analysis_id),
            ("provider", self.provider),
            ("model", self.model),
            ("error", self.error),
        ):
            _validate_optional_string(value, field_name)

        # --------------------------------------------------
        # DURAÇÃO
        # --------------------------------------------------

        if self.duration_ms is not None:
            if (
                isinstance(self.duration_ms, bool)
                or not isinstance(self.duration_ms, (int, float))
                or not isfinite(self.duration_ms)
                or self.duration_ms < 0
            ):
                raise ValueError(
                    "duration_ms deve ser um número finito não negativo ou None."
                )

        # --------------------------------------------------
        # CACHE
        # --------------------------------------------------

        if self.cache_hit is not None and not isinstance(self.cache_hit, bool):
            raise TypeError("cache_hit deve ser um booleano ou None.")

        # --------------------------------------------------
        # METADADOS
        # --------------------------------------------------

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata deve ser um dicionário.")

        # --------------------------------------------------
        # ESTIMATIVA DE CUSTO
        # --------------------------------------------------

        if self.token_cost_estimate is not None and not isinstance(
            self.token_cost_estimate,
            dict,
        ):
            raise TypeError("token_cost_estimate deve ser um dicionário ou None.")

    @property
    def passed(self) -> bool:
        """Indica se o cenário foi aprovado."""

        return self.status == "passed"

    @property
    def failed(self) -> bool:
        """Indica se a comparação foi reprovada."""

        return self.status == "failed"

    @property
    def estimated_cost_usd(self) -> float | None:
        """Retorna o custo estimado do cenário, quando disponível."""

        if self.token_cost_estimate is None:
            return None

        return _non_negative_number(self.token_cost_estimate.get("estimated_cost_usd"))


# ==========================================================
# RELATÓRIO CONSOLIDADO
# ==========================================================


@dataclass(slots=True)
class EvaluationRunReport:
    """Representa um relatório consolidado de avaliação."""

    run_id: str

    provider: str | None = None
    model: str | None = None

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    finished_at: datetime | None = None

    results: list[EvaluationCaseResult] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida a estrutura do relatório."""

        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id deve ser uma string não vazia.")

        _validate_optional_string(self.provider, "provider")

        _validate_optional_string(self.model, "model")

        self.started_at = _normalize_datetime(
            self.started_at,
            "started_at",
        )

        if self.finished_at is not None:
            self.finished_at = _normalize_datetime(
                self.finished_at,
                "finished_at",
            )

        if not isinstance(self.results, list):
            raise TypeError("results deve ser uma lista de EvaluationCaseResult.")

        if any(not isinstance(result, EvaluationCaseResult) for result in self.results):
            raise TypeError(
                "Todos os elementos de results devem ser EvaluationCaseResult."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata deve ser um dicionário.")

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.status == "passed" for result in self.results)

    @property
    def failed_cases(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    @property
    def error_cases(self) -> int:
        return sum(result.status == "error" for result in self.results)

    @property
    def skipped_cases(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def evaluated_cases(self) -> int:
        """Quantidade de cenários efetivamente avaliados."""

        return self.passed_cases + self.failed_cases + self.error_cases

    @property
    def pass_rate(self) -> float:
        """Taxa de aprovação em percentual.

        Cenários ignorados não entram no denominador.
        Erros de execução entram como reprovação.
        """

        if self.evaluated_cases == 0:
            return 0.0

        return round(
            self.passed_cases / self.evaluated_cases * 100,
            2,
        )

    @property
    def duration_ms(self) -> float:
        """Duração total da avaliação em milissegundos."""

        end_time = self.finished_at or datetime.now(timezone.utc)

        # started_at e finished_at são normalizados para UTC
        # quando chegam sem informação de fuso horário.
        duration = (end_time - self.started_at).total_seconds() * 1000

        return round(max(duration, 0.0), 2)

    @property
    def token_cost_summary(self) -> dict[str, Any]:
        """Retorna custos agregados por cenário, modelo e provedor."""

        return _aggregate_token_costs(self.results)

    @property
    def estimated_cost_usd(self) -> float | None:
        """Retorna o custo total estimado da execução."""

        return self.token_cost_summary["estimated_cost_usd"]

    def finalize(self) -> None:
        """Registra o horário de conclusão da avaliação."""

        if self.finished_at is not None:
            raise RuntimeError("Este relatório já foi finalizado.")

        self.finished_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Converte o relatório para uma estrutura serializável.

        Os metadados são copiados para que alterações no dicionário
        retornado não modifiquem o relatório original.
        """

        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at.isoformat(),
            "finished_at": (self.finished_at.isoformat() if self.finished_at else None),
            "summary": {
                "total_cases": self.total_cases,
                "evaluated_cases": self.evaluated_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "error_cases": self.error_cases,
                "skipped_cases": self.skipped_cases,
                "pass_rate": self.pass_rate,
                "duration_ms": self.duration_ms,
            },
            "token_cost_summary": self.token_cost_summary,
            "results": [asdict(result) for result in self.results],
            "metadata": deepcopy(self.metadata),
        }
