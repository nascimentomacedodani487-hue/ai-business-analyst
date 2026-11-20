"""Rotas REST para consulta e comparação de benchmarks históricos."""

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ai_analyst.evaluation.comparison import BenchmarkComparator
from ai_analyst.evaluation.history import BenchmarkHistory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Evaluation"])

# Utiliza exatamente o banco padrão compartilhado com a CLI.
benchmark_history = BenchmarkHistory()

# Centraliza as regras de comparação no BenchmarkComparator existente.
benchmark_comparator = BenchmarkComparator()


class BenchmarkComparisonRequest(BaseModel):
    """Identificadores dos benchmarks que serão comparados."""

    baseline_id: str = Field(min_length=1, max_length=150)
    candidate_id: str = Field(min_length=1, max_length=150)


_PRIVATE_ERROR_FIELDS = {
    "error",
    "error_message",
    "exception_message",
    "traceback",
    "stack_trace",
}

SAFE_ERROR_DETAIL = "Detalhes internos omitidos por segurança."


def _sanitize_public_value(value: Any) -> Any:
    """Remove mensagens internas de erro dos relatórios expostos pela API.

    Mantém os tipos de erro e os demais metadados necessários
    para compreender o resultado da execução.
    """

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}

        for key, item in value.items():
            normalized_key = str(key).lower()

            if (
                normalized_key in _PRIVATE_ERROR_FIELDS
                and item is not None
                and item != ""
            ):
                sanitized[key] = SAFE_ERROR_DETAIL
            else:
                sanitized[key] = _sanitize_public_value(item)

        return sanitized

    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]

    return value


def _load_benchmark(
    benchmark_id: str,
    *,
    label: str = "solicitado",
) -> dict[str, Any]:
    """Recupera um benchmark ou produz uma resposta HTTP controlada."""

    try:
        report = benchmark_history.get_benchmark(benchmark_id)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Erro ao recuperar benchmark histórico.",
            extra={
                "event": "evaluation_benchmark_retrieval_failed",
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Não foi possível recuperar o benchmark.",
        ) from exc

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark {label} não encontrado.",
        )

    return report


@router.get("/evaluation/benchmarks")
def list_evaluation_benchmarks(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    provider: str | None = Query(default=None),
    model: str | None = Query(default=None),
    dataset_sha256: str | None = Query(default=None),
):
    """Lista benchmarks históricos com filtros e paginação."""

    try:
        return benchmark_history.list_benchmarks(
            limit=limit,
            offset=offset,
            provider=provider,
            model=model,
            dataset_sha256=dataset_sha256,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Erro ao consultar o histórico de benchmarks.",
            extra={
                "event": "evaluation_benchmark_list_failed",
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Não foi possível consultar o histórico de benchmarks.",
        ) from exc


@router.post("/evaluation/benchmarks/compare")
def compare_evaluation_benchmarks(
    request: BenchmarkComparisonRequest,
):
    """Compara dois benchmarks já persistidos.

    Esta operação não executa agentes nem faz chamadas a provedores LLM.
    """

    baseline_id = request.baseline_id.strip()
    candidate_id = request.candidate_id.strip()

    if not baseline_id or not candidate_id:
        raise HTTPException(
            status_code=400,
            detail="Os identificadores dos benchmarks não podem ser vazios.",
        )

    if baseline_id == candidate_id:
        raise HTTPException(
            status_code=400,
            detail="Os benchmarks de referência e candidato devem ser diferentes.",
        )

    baseline = _load_benchmark(
        baseline_id,
        label="de referência",
    )

    candidate = _load_benchmark(
        candidate_id,
        label="candidato",
    )

    try:
        return benchmark_comparator.compare(
            baseline=baseline,
            candidate=candidate,
        )

    except Exception as exc:
        logger.exception(
            "Erro ao comparar benchmarks históricos.",
            extra={
                "event": "evaluation_benchmark_comparison_failed",
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Não foi possível comparar os benchmarks.",
        ) from exc


@router.get("/evaluation/benchmarks/{benchmark_id}")
def get_evaluation_benchmark(benchmark_id: str):
    """Recupera os detalhes completos de um benchmark."""

    report = _load_benchmark(benchmark_id)

    return _sanitize_public_value(report)
