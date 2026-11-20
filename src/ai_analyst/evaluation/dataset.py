"""Carregamento e validação do dataset de avaliação.

Suporta cenários de avaliação SQL e comportamental:
- sql_result
- clarification
- unanswerable
- partial_answer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_analyst.evaluation.models import EvaluationCase

DATASET_VERSION = 1

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "evaluation_dataset.json"
)

# ==========================================================
# CAMPOS PERMITIDOS
# ==========================================================

ALLOWED_CASE_FIELDS = {
    "case_id",
    "question",
    "reference_sql",
    "category",
    "description",
    "comparison_mode",
    "absolute_tolerance",
    "relative_tolerance",
    "required_columns",
    "allow_extra_columns",
    "sort_rows",
    "tags",
    "expected_analysis_mode",
    "evaluation_type",
    "expected_behavior",
    "evaluation_criteria",
}

# Campos representados como tuplas no EvaluationCase,
# mas como listas no arquivo JSON.
TUPLE_FIELDS = (
    "required_columns",
    "tags",
    "evaluation_criteria",
)


# ==========================================================
# NORMALIZAÇÃO DOS CENÁRIOS
# ==========================================================


def normalize_case_data(
    raw_case: dict[str, Any],
    position: int,
) -> dict[str, Any]:
    """Valida e normaliza os campos de um cenário JSON.

    Converte listas em tuplas e permite omitir reference_sql
    exclusivamente nos cenários comportamentais.

    Args:
        raw_case: Objeto JSON do cenário.
        position: Posição do cenário no dataset.

    Returns:
        Dicionário preparado para construir EvaluationCase.

    Raises:
        ValueError: Quando a estrutura do cenário é inválida.
    """

    case_data = dict(raw_case)

    # ------------------------------------------------------
    # CAMPOS DESCONHECIDOS
    # ------------------------------------------------------

    unknown_fields = set(case_data) - ALLOWED_CASE_FIELDS

    if unknown_fields:
        fields = ", ".join(sorted(unknown_fields))

        raise ValueError(f"Campos desconhecidos no cenário {position}: {fields}.")

    # ------------------------------------------------------
    # MODALIDADE DE AVALIAÇÃO
    # ------------------------------------------------------

    evaluation_type = case_data.get(
        "evaluation_type",
        "sql_result",
    )

    # ------------------------------------------------------
    # SQL DE REFERÊNCIA
    # ------------------------------------------------------

    # O SQL continua obrigatório para cenários tradicionais.
    #
    # Cenários comportamentais não precisam de SQL de referência.
    # Quando omitido, o carregador define explicitamente None,
    # mantendo compatibilidade com a assinatura de EvaluationCase.

    if "reference_sql" not in case_data and evaluation_type != "sql_result":
        case_data["reference_sql"] = None

    # ------------------------------------------------------
    # CAMPOS REPRESENTADOS COMO TUPLAS
    # ------------------------------------------------------

    for field_name in TUPLE_FIELDS:
        value = case_data.get(field_name, [])

        if not isinstance(value, list):
            raise ValueError(
                f"'{field_name}' deve ser uma lista no cenário {position}."
            )

        if not all(isinstance(item, str) for item in value):
            raise ValueError(
                f"Todos os elementos de '{field_name}' "
                f"devem ser strings no cenário {position}."
            )

        case_data[field_name] = tuple(value)

    return case_data


# ==========================================================
# CARREGAMENTO DO DATASET
# ==========================================================


def load_evaluation_dataset(
    path: str | Path | None = None,
) -> list[EvaluationCase]:
    """Carrega e valida os cenários de avaliação de um arquivo JSON.

    O carregador oferece compatibilidade com os cenários SQL
    existentes e suporta as novas modalidades comportamentais.

    Modalidades suportadas:

        sql_result:
            Exige reference_sql.

        clarification:
            Espera que o agente solicite esclarecimentos.

        unanswerable:
            Espera que o agente reconheça que não pode responder
            com os dados disponíveis.

        partial_answer:
            Espera que o agente diferencie informações disponíveis
            de informações que não podem ser determinadas.

    Args:
        path: Caminho opcional do dataset.
            Quando omitido, utiliza DEFAULT_DATASET_PATH.

    Returns:
        Lista de cenários representados por EvaluationCase.

    Raises:
        FileNotFoundError:
            Quando o arquivo não existe.

        ValueError:
            Quando o JSON, sua estrutura ou algum cenário são inválidos.
    """

    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset de avaliação não encontrado: {dataset_path}")

    # ------------------------------------------------------
    # LEITURA DO JSON
    # ------------------------------------------------------

    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido no dataset '{dataset_path}': {exc}") from exc

    # ------------------------------------------------------
    # VALIDAÇÃO DA RAIZ
    # ------------------------------------------------------

    if not isinstance(payload, dict):
        raise ValueError("A raiz do dataset deve ser um objeto JSON.")

    # ------------------------------------------------------
    # VERSÃO DO DATASET
    # ------------------------------------------------------

    version = payload.get("version")

    if version != DATASET_VERSION:
        raise ValueError(
            f"Versão de dataset não suportada: {version}. "
            f"Versão esperada: {DATASET_VERSION}."
        )

    # ------------------------------------------------------
    # LISTA DE CENÁRIOS
    # ------------------------------------------------------

    raw_cases = payload.get("cases")

    if not isinstance(raw_cases, list):
        raise ValueError("O campo 'cases' deve conter uma lista.")

    # ------------------------------------------------------
    # CONSTRUÇÃO DOS CENÁRIOS
    # ------------------------------------------------------

    cases: list[EvaluationCase] = []

    seen_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        position = index + 1

        # --------------------------------------------------
        # VALIDAR FORMATO DO CENÁRIO
        # --------------------------------------------------

        if not isinstance(raw_case, dict):
            raise ValueError(
                f"O cenário na posição {position} deve ser um objeto JSON."
            )

        # --------------------------------------------------
        # NORMALIZAR CAMPOS
        # --------------------------------------------------

        case_data = normalize_case_data(
            raw_case=raw_case,
            position=position,
        )

        # --------------------------------------------------
        # CONSTRUIR MODELO TIPADO
        # --------------------------------------------------

        try:
            case = EvaluationCase(**case_data)

        except (TypeError, ValueError) as exc:
            raise ValueError(f"Erro no cenário da posição {position}: {exc}") from exc

        # --------------------------------------------------
        # VALIDAR IDENTIFICADOR
        # --------------------------------------------------

        normalized_id = case.case_id.strip()

        if normalized_id in seen_ids:
            raise ValueError(f"ID de cenário duplicado: '{case.case_id}'.")

        seen_ids.add(normalized_id)

        cases.append(case)

    return cases
