from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_analyst.evaluation.dataset import (
    DATASET_VERSION,
    load_evaluation_dataset,
)

# ==========================================================
# FÁBRICAS DE DATASET
# ==========================================================


def make_dataset_case(
    **overrides: Any,
) -> dict[str, Any]:
    """Cria um cenário mínimo válido para o dataset."""

    case = {
        "case_id": "test_case",
        "question": "Quantos pedidos existem?",
        "reference_sql": "SELECT COUNT(*) AS total_orders FROM orders",
        "category": "volume",
    }

    case.update(overrides)

    return case


def write_dataset(
    path: Path,
    cases: list[dict[str, Any]],
    version: int = DATASET_VERSION,
) -> Path:
    """Escreve um dataset temporário em JSON."""

    payload = {
        "version": version,
        "cases": cases,
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


# ==========================================================
# CARREGAMENTO DO MODO DE ANÁLISE
# ==========================================================


@pytest.mark.parametrize(
    "analysis_mode",
    [
        "single_query",
        "multi_query",
    ],
)
def test_dataset_loads_expected_analysis_mode(
    tmp_path: Path,
    analysis_mode: str,
):
    """Carrega corretamente os modos de análise suportados."""

    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(
                expected_analysis_mode=analysis_mode,
            ),
        ],
    )

    cases = load_evaluation_dataset(dataset_path)

    assert len(cases) == 1

    assert cases[0].expected_analysis_mode == analysis_mode


def test_dataset_defaults_expected_analysis_mode_to_none(
    tmp_path: Path,
):
    """Preserva a compatibilidade com datasets antigos."""

    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(),
        ],
    )

    cases = load_evaluation_dataset(dataset_path)

    assert len(cases) == 1

    assert cases[0].expected_analysis_mode is None


def test_dataset_accepts_explicit_null_analysis_mode(
    tmp_path: Path,
):
    """Aceita null explicitamente no JSON."""

    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(
                expected_analysis_mode=None,
            ),
        ],
    )

    cases = load_evaluation_dataset(dataset_path)

    assert len(cases) == 1

    assert cases[0].expected_analysis_mode is None


# ==========================================================
# VALIDAÇÃO DO MODO DE ANÁLISE
# ==========================================================


@pytest.mark.parametrize(
    "invalid_mode",
    [
        "multiple_query",
        "invalid",
        "MULTI_QUERY",
        "",
        123,
        [],
        {},
    ],
)
def test_dataset_rejects_invalid_analysis_modes(
    tmp_path: Path,
    invalid_mode: Any,
):
    """Rejeita modos que não são suportados pelo modelo."""

    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(
                expected_analysis_mode=invalid_mode,
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="expected_analysis_mode deve ser None",
    ):
        load_evaluation_dataset(dataset_path)


# ==========================================================
# MÚLTIPLOS CENÁRIOS
# ==========================================================


def test_dataset_preserves_different_analysis_modes(
    tmp_path: Path,
):
    """Preserva a expectativa individual de cada cenário."""

    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(
                case_id="simple_analysis",
                expected_analysis_mode="single_query",
            ),
            make_dataset_case(
                case_id="composite_analysis",
                expected_analysis_mode="multi_query",
            ),
            make_dataset_case(
                case_id="unrestricted_analysis",
            ),
        ],
    )

    cases = load_evaluation_dataset(dataset_path)

    assert len(cases) == 3

    modes_by_id = {case.case_id: case.expected_analysis_mode for case in cases}

    assert modes_by_id == {
        "simple_analysis": "single_query",
        "composite_analysis": "multi_query",
        "unrestricted_analysis": None,
    }


# ==========================================================
# COMPATIBILIDADE E VALIDAÇÃO DO DATASET
# ==========================================================


def test_dataset_rejects_unsupported_version(
    tmp_path: Path,
):
    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(),
        ],
        version=999,
    )

    with pytest.raises(
        ValueError,
        match="Versão de dataset não suportada",
    ):
        load_evaluation_dataset(dataset_path)


def test_dataset_rejects_unknown_fields(
    tmp_path: Path,
):
    dataset_path = write_dataset(
        tmp_path / "evaluation_dataset.json",
        [
            make_dataset_case(
                unsupported_field="invalid",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Campos desconhecidos",
    ):
        load_evaluation_dataset(dataset_path)
