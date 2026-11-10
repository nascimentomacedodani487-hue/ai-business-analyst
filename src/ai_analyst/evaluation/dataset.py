"""Carregamento e validação do dataset de avaliação."""

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


def load_evaluation_dataset(
    path: str | Path | None = None,
) -> list[EvaluationCase]:
    """Carrega e valida os cenários de avaliação de um arquivo JSON.

    Args:
        path: Caminho opcional do dataset.
            Quando omitido, utiliza DEFAULT_DATASET_PATH.

    Returns:
        Lista de cenários representados por EvaluationCase.

    Raises:
        FileNotFoundError: Quando o arquivo não existe.
        ValueError: Quando o JSON ou sua estrutura são inválidos.
    """

    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset de avaliação não encontrado: {dataset_path}")

    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido no dataset '{dataset_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("A raiz do dataset deve ser um objeto JSON.")

    version = payload.get("version")

    if version != DATASET_VERSION:
        raise ValueError(
            f"Versão de dataset não suportada: {version}. "
            f"Versão esperada: {DATASET_VERSION}."
        )

    raw_cases = payload.get("cases")

    if not isinstance(raw_cases, list):
        raise ValueError("O campo 'cases' deve conter uma lista.")

    allowed_fields = {
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
    }

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()

    for index, raw_case in enumerate(raw_cases):
        position = index + 1

        if not isinstance(raw_case, dict):
            raise ValueError(
                f"O cenário na posição {position} deve ser um objeto JSON."
            )

        case_data: dict[str, Any] = dict(raw_case)

        unknown_fields = set(case_data) - allowed_fields

        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))

            raise ValueError(f"Campos desconhecidos no cenário {position}: {fields}.")

        for field_name in ("required_columns", "tags"):
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

        try:
            case = EvaluationCase(**case_data)

        except (TypeError, ValueError) as exc:
            raise ValueError(f"Erro no cenário da posição {position}: {exc}") from exc

        normalized_id = case.case_id.strip()

        if normalized_id in seen_ids:
            raise ValueError(f"ID de cenário duplicado: '{case.case_id}'.")

        seen_ids.add(normalized_id)
        cases.append(case)

    return cases
