"""Interface de linha de comando do AI Evaluation Engine.

Modos disponíveis:
- Listagem dos cenários.
- Validação dos SQLs de referência sem chamadas de IA.
- Execução individual com autorização explícita para utilizar LLMs.
- Benchmark comparativo entre múltiplos provedores.
- Consulta do histórico de benchmarks.
- Comparação histórica entre duas execuções.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence
from uuid import uuid4

from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.evaluation.comparison import BenchmarkComparator
from ai_analyst.evaluation.dataset import load_evaluation_dataset
from ai_analyst.evaluation.history import BenchmarkHistory

logger = logging.getLogger(__name__)


SUPPORTED_PROVIDERS = (
    "gemini",
    "openai",
    "anthropic",
)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ERROR = 2


# ==========================================================
# ARGUMENTOS
# ==========================================================


def positive_int(value: str) -> int:
    """Valida um inteiro estritamente positivo."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("O valor deve ser um número inteiro.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("O valor deve ser maior que zero.")

    return parsed


def non_negative_int(value: str) -> int:
    """Valida um inteiro maior ou igual a zero."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("O valor deve ser um número inteiro.") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("O valor não pode ser negativo.")

    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos da CLI."""

    parser = argparse.ArgumentParser(
        prog="ai-evaluation",
        description=(
            "Executa avaliações do AI Business Analyst "
            "com métricas, comparação determinística e relatórios."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  ai-evaluation --list-cases\n"
            "  ai-evaluation --validate-only\n"
            "  ai-evaluation --validate-only --case total_orders\n"
            "  ai-evaluation --provider gemini --allow-llm\n"
            "  ai-evaluation --provider openai --allow-llm "
            "--output reports/openai.json\n"
            "  ai-evaluation --benchmark gemini openai anthropic "
            "--allow-llm --output reports/benchmark.json\n"
            "  ai-evaluation --list-benchmarks\n"
            "  ai-evaluation --list-benchmarks "
            "--filter-provider gemini --limit 10\n"
            "  ai-evaluation --show-benchmark benchmark-id\n"
            "  ai-evaluation --compare-benchmarks "
            "baseline-id candidate-id"
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--list-cases",
        action="store_true",
        help="Lista os cenários disponíveis sem executar avaliações.",
    )

    mode_group.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Executa apenas os SQLs de referência e os guardrails. "
            "Não utiliza provedores de IA."
        ),
    )

    mode_group.add_argument(
        "--benchmark",
        nargs="+",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        metavar="PROVIDER",
        help=(
            "Executa um benchmark comparativo entre pelo menos "
            "dois provedores. Exemplo: --benchmark gemini openai anthropic."
        ),
    )

    mode_group.add_argument(
        "--list-benchmarks",
        action="store_true",
        help=(
            "Lista os benchmarks armazenados no histórico SQLite "
            "sem executar provedores de IA."
        ),
    )

    mode_group.add_argument(
        "--show-benchmark",
        type=str,
        default=None,
        metavar="BENCHMARK_ID",
        help=(
            "Exibe o relatório completo de um benchmark armazenado "
            "pelo identificador informado."
        ),
    )

    mode_group.add_argument(
        "--compare-benchmarks",
        nargs=2,
        default=None,
        metavar=("BASELINE_ID", "CANDIDATE_ID"),
        help=(
            "Compara dois benchmarks históricos pelo identificador, "
            "detectando regressões e melhorias sem executar LLMs."
        ),
    )

    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default="gemini",
        help=(
            "Provedor de IA utilizado na avaliação individual. "
            "Não é utilizado quando --benchmark é informado."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Caminho alternativo para o dataset de avaliação.",
    )

    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=None,
        metavar="CASE_ID",
        help=(
            "Identificador de um cenário específico. Pode ser informado várias vezes."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Caminho do arquivo JSON que receberá o relatório ou resultado.",
    )

    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help=(
            "Autoriza explicitamente chamadas reais aos provedores "
            "de IA. Obrigatório para avaliações completas e benchmarks."
        ),
    )

    # ------------------------------------------------------
    # Consulta do histórico de benchmarks
    # ------------------------------------------------------

    parser.add_argument(
        "--filter-provider",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        metavar="PROVIDER",
        help=("Filtra o histórico pelo provedor. Utilizado com --list-benchmarks."),
    )

    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "Filtra o histórico pelo modelo utilizado. Utilizado com --list-benchmarks."
        ),
    )

    parser.add_argument(
        "--limit",
        type=positive_int,
        default=20,
        metavar="N",
        help="Quantidade máxima de registros exibidos na listagem do histórico.",
    )

    parser.add_argument(
        "--offset",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="Quantidade de registros ignorados antes da listagem do histórico.",
    )

    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        help="Nível de detalhamento dos logs.",
    )

    return parser


# ==========================================================
# LOGGING
# ==========================================================


def configure_logging(level: str) -> None:
    """Configura o sistema de logs da CLI."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


# ==========================================================
# CARREGAMENTO E SELEÇÃO DOS CENÁRIOS
# ==========================================================


def load_cases(
    dataset_path: Path | None = None,
):
    """Carrega o dataset de avaliação."""

    cases = load_evaluation_dataset(dataset_path)

    if not cases:
        raise ValueError("O dataset não contém cenários de avaliação.")

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("O dataset contém identificadores de cenários duplicados.")

    return cases


def select_cases(
    cases: list,
    requested_ids: list[str] | None,
) -> list:
    """Filtra os cenários solicitados pela linha de comando."""

    if not requested_ids:
        return cases

    available_ids = {case.case_id for case in cases}

    requested_set = set(requested_ids)

    unknown_ids = sorted(requested_set - available_ids)

    if unknown_ids:
        raise ValueError(
            "Cenário(s) não encontrado(s): "
            + ", ".join(unknown_ids)
            + ". Utilize --list-cases para consultar os IDs disponíveis."
        )

    return [case for case in cases if case.case_id in requested_set]


# ==========================================================
# LISTAGEM DOS CENÁRIOS
# ==========================================================


def list_cases(cases: list) -> None:
    """Exibe os cenários disponíveis."""

    print()
    print("AI BUSINESS ANALYST — EVALUATION DATASET")
    print("=" * 72)

    print(f"Total de cenários: {len(cases)}")
    print()

    for index, case in enumerate(cases, start=1):
        print(f"{index:02d}. {case.case_id}")

        print(f"    Categoria: {case.category}")

        print(f"    Comparação: {case.comparison_mode}")

        if case.expected_analysis_mode is not None:
            print(f"    Modo esperado: {case.expected_analysis_mode}")

        print(f"    Pergunta: {case.question}")

        if case.description:
            print(f"    Descrição: {case.description}")

        print()


# ==========================================================
# SERIALIZAÇÃO JSON
# ==========================================================


def json_default(value: Any) -> Any:
    """Converte tipos adicionais para formatos compatíveis com JSON."""

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    raise TypeError(
        f"Tipo não suportado para serialização JSON: {type(value).__name__}."
    )


def write_json_report(
    output_path: Path,
    payload: dict[str, Any],
) -> Path:
    """Salva um relatório JSON de maneira atômica."""

    output_path = output_path.expanduser()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")

    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=json_default,
            allow_nan=False,
        )

        temporary_path.write_text(
            serialized + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path


# ==========================================================
# CONSULTA DO HISTÓRICO DE BENCHMARKS
# ==========================================================


def _format_history_list(values: Any) -> str:
    """Formata listas ou mapeamentos de valores para exibição."""

    if isinstance(values, Mapping):
        normalized = [
            str(value)
            for value in values.values()
            if value is not None and str(value).strip()
        ]

    elif isinstance(values, (list, tuple, set)):
        normalized = [
            str(value) for value in values if value is not None and str(value).strip()
        ]

    else:
        return "N/D"

    return ", ".join(normalized) if normalized else "N/D"


def print_benchmark_history(
    entries: list[dict[str, Any]],
    total: int,
    limit: int,
    offset: int,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    """Exibe os registros resumidos do histórico de benchmarks."""

    print()
    print("AI BUSINESS ANALYST — HISTÓRICO DE BENCHMARKS")
    print("=" * 88)

    if provider:
        print(f"Filtro de provedor: {provider}")

    if model:
        print(f"Filtro de modelo: {model}")

    print(f"Registros encontrados: {total}")

    print(f"Registros exibidos: {len(entries)}")

    print(f"Paginação: offset={offset}, limit={limit}")

    if not entries:
        print()
        print("Nenhum benchmark encontrado no histórico.")
        print()

        return

    print()
    print("-" * 88)

    for index, entry in enumerate(entries, start=offset + 1):
        dataset = entry.get("dataset", {})
        summary = entry.get("summary", {})

        if not isinstance(dataset, Mapping):
            dataset = {}

        if not isinstance(summary, Mapping):
            summary = {}

        benchmark_id = entry.get(
            "benchmark_id",
            "N/D",
        )

        created_at = entry.get(
            "created_at",
            "N/D",
        )

        dataset_case_count = dataset.get(
            "case_count",
            entry.get("dataset_case_count", entry.get("case_count", "N/D")),
        )

        dataset_sha256 = dataset.get(
            "sha256",
            entry.get("dataset_sha256", entry.get("sha256", "N/D")),
        )

        providers = _format_history_list(
            entry.get("providers"),
        )

        models = _format_history_list(
            entry.get("models"),
        )

        provider_count = summary.get(
            "provider_count",
            entry.get("provider_count", "N/D"),
        )

        completed_providers = summary.get(
            "completed_providers",
            entry.get("completed_providers", "N/D"),
        )

        failed_providers = summary.get(
            "failed_providers",
            entry.get("failed_providers", "N/D"),
        )

        print(f"{index:03d}. {benchmark_id}")

        print(f"     Criado em: {created_at}")

        print(f"     Cenários: {dataset_case_count}")

        print(f"     Provedores: {providers}")

        print(f"     Modelos: {models}")

        print(
            f"     Execuções: {completed_providers} concluídas / "
            f"{failed_providers} com falha "
            f"(total de provedores: {provider_count})"
        )

        print(f"     Dataset SHA-256: {dataset_sha256}")

        print("-" * 88)

    print()


def list_benchmark_history(
    history_store: BenchmarkHistory,
    limit: int,
    offset: int,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Recupera uma página do histórico e suas informações de paginação."""

    result = history_store.list_benchmarks(
        limit=limit,
        offset=offset,
        provider=provider,
        model=model,
    )

    if isinstance(result, Mapping):
        entries = result.get("items")

        pagination = result.get("pagination")

        if not isinstance(entries, list):
            raise ValueError("O histórico retornou um campo 'items' inválido.")

        if not isinstance(pagination, Mapping):
            raise ValueError("O histórico retornou um campo 'pagination' inválido.")

        total = pagination.get("total")
        stored_limit = pagination.get("limit")
        stored_offset = pagination.get("offset")
        has_more = pagination.get("has_more")

        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError("O total de registros do histórico é inválido.")

        if (
            not isinstance(stored_limit, int)
            or isinstance(stored_limit, bool)
            or stored_limit <= 0
        ):
            raise ValueError(
                "O limite de paginação retornado pelo histórico é inválido."
            )

        if (
            not isinstance(stored_offset, int)
            or isinstance(stored_offset, bool)
            or stored_offset < 0
        ):
            raise ValueError(
                "O offset de paginação retornado pelo histórico é inválido."
            )

        if not isinstance(has_more, bool):
            raise ValueError(
                "O indicador has_more retornado pelo histórico é inválido."
            )

        normalized_pagination = {
            "total": total,
            "limit": stored_limit,
            "offset": stored_offset,
            "has_more": has_more,
        }

    elif isinstance(result, list):
        entries = result

        total = history_store.count_benchmarks(
            provider=provider,
            model=model,
        )

        normalized_pagination = {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(entries) < total,
        }

    else:
        raise ValueError("O histórico retornou um formato de listagem não suportado.")

    if not all(isinstance(entry, Mapping) for entry in entries):
        raise ValueError("O histórico retornou registros em formato inválido.")

    return {
        "items": entries,
        "pagination": normalized_pagination,
        "filters": {
            "provider": provider,
            "model": model,
        },
    }


def show_benchmark_history(
    history_store: BenchmarkHistory,
    benchmark_id: str,
) -> dict[str, Any] | None:
    """Recupera o relatório completo de um benchmark pelo identificador."""

    return history_store.get_benchmark(benchmark_id)


# ==========================================================
# COMPARAÇÃO HISTÓRICA DE BENCHMARKS
# ==========================================================


def execute_benchmark_comparison(
    history_store: BenchmarkHistory,
    baseline_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Recupera e compara dois benchmarks históricos.

    Não executa provedores de IA. Utiliza exclusivamente os relatórios
    já persistidos no histórico SQLite.

    Raises:
        ValueError: quando algum benchmark não existe.
    """

    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise ValueError("O identificador do benchmark baseline é inválido.")

    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("O identificador do benchmark candidate é inválido.")

    if baseline_id == candidate_id:
        raise ValueError(
            "Os identificadores baseline e candidate devem ser diferentes."
        )

    baseline = history_store.get_benchmark(baseline_id)

    if baseline is None:
        raise ValueError(f"Benchmark baseline não encontrado: {baseline_id}")

    candidate = history_store.get_benchmark(candidate_id)

    if candidate is None:
        raise ValueError(f"Benchmark candidate não encontrado: {candidate_id}")

    comparator = BenchmarkComparator()

    return comparator.compare(
        baseline=baseline,
        candidate=candidate,
    )


def benchmark_comparison_exit_code(
    payload: Any,
) -> int:
    """Determina o código de saída de uma comparação histórica.

    Comparações estáveis ou com melhoria retornam sucesso.
    Regressões, resultados mistos e incompatibilidades retornam falha.
    """

    if not isinstance(payload, Mapping):
        return EXIT_FAILURE

    summary = payload.get("summary")

    if not isinstance(summary, Mapping):
        return EXIT_FAILURE

    status = summary.get("status")

    if status in {"stable", "improvement"}:
        return EXIT_SUCCESS

    return EXIT_FAILURE


def _format_comparison_value(
    value: Any,
    metric_name: str,
) -> str:
    """Formata um valor de métrica para o relatório."""

    if value is None:
        return "N/D"

    if isinstance(value, bool):
        return str(value)

    if not isinstance(value, (int, float)):
        return str(value)

    if metric_name in {
        "pass_rate",
        "row_accuracy",
        "cell_accuracy",
    }:
        return f"{value:.2f}%"

    if metric_name == "average_score":
        return f"{value:.4f}"

    if isinstance(value, int):
        return str(value)

    return f"{value:.2f}"


def _comparison_metric_label(metric_name: str) -> str:
    """Converte identificadores técnicos em nomes legíveis."""

    labels = {
        "pass_rate": "Taxa de aprovação",
        "average_score": "Score médio",
        "average_duration_ms": "Latência média por análise",
        "row_accuracy": "Row accuracy",
        "cell_accuracy": "Cell accuracy",
        "error_cases": "Cenários com erro",
        "failed_cases": "Cenários reprovados",
        "skipped_cases": "Cenários ignorados",
        "llm_usage.tokens.input": "Tokens de entrada",
        "llm_usage.tokens.output": "Tokens de saída",
        "llm_usage.tokens.thinking": "Tokens de raciocínio",
        "llm_usage.tokens.total": "Total de tokens",
        "llm_usage.tokens.unreported": "Tokens não reportados",
        "llm_usage.requests.failed": "Requisições com falha",
        "llm_usage.requests.retries": "Tentativas adicionais",
        "llm_usage.latency.average_request_ms": "Latência média por requisição",
    }

    return labels.get(metric_name, metric_name)


def print_benchmark_comparison_report(
    payload: dict[str, Any],
) -> None:
    """Exibe o resultado comparativo de dois benchmarks."""

    baseline = payload.get("baseline", {})
    candidate = payload.get("candidate", {})
    compatibility = payload.get("compatibility", {})
    summary = payload.get("summary", {})
    providers = payload.get("providers", {})

    if not isinstance(baseline, Mapping):
        baseline = {}

    if not isinstance(candidate, Mapping):
        candidate = {}

    if not isinstance(compatibility, Mapping):
        compatibility = {}

    if not isinstance(summary, Mapping):
        summary = {}

    if not isinstance(providers, Mapping):
        providers = {}

    print()
    print("AI BUSINESS ANALYST — BENCHMARK COMPARISON")
    print("=" * 88)

    print(f"Baseline: {baseline.get('benchmark_id', 'N/D')}")

    print(f"Candidate: {candidate.get('benchmark_id', 'N/D')}")

    print(f"Dataset SHA-256: {baseline.get('dataset_sha256', 'N/D')}")

    print(f"Cenários: {baseline.get('dataset_case_count', 'N/D')}")

    print()

    status = str(summary.get("status", "not_comparable")).upper()

    print(f"RESULTADO GERAL: {status}")

    print(
        f"Provedores comparados: "
        f"{summary.get('providers_compared', 0)}"
        f"/{summary.get('providers_total', 0)}"
    )

    print(f"Regressões: {summary.get('regressions', 0)}")

    print(f"Melhorias: {summary.get('improvements', 0)}")

    print(f"Métricas estáveis: {summary.get('stable_metrics', 0)}")

    print(f"Métricas não comparáveis: {summary.get('not_comparable_metrics', 0)}")

    if not compatibility.get("comparable", False):
        print()
        print("COMPATIBILIDADE: INCOMPATIBLE")

        reasons = compatibility.get("reasons", [])

        if isinstance(reasons, list):
            for reason in reasons:
                print(f"  - {reason}")

        print()
        return

    print()
    print("-" * 88)
    print("COMPARAÇÃO POR PROVEDOR")
    print("-" * 88)

    for provider, provider_result in providers.items():
        if not isinstance(provider_result, Mapping):
            continue

        print()
        print(f"PROVEDOR: {provider.upper()}")

        provider_status = str(provider_result.get("status", "not_comparable")).upper()

        print(f"  Status: {provider_status}")

        print(f"  Modelo baseline: {provider_result.get('baseline_model') or 'N/D'}")

        print(f"  Modelo candidate: {provider_result.get('candidate_model') or 'N/D'}")

        execution = provider_result.get("execution")

        if isinstance(execution, Mapping):
            print(
                f"  Execução: "
                f"{execution.get('baseline', 'N/D')} -> "
                f"{execution.get('candidate', 'N/D')}"
            )

        metrics = provider_result.get("metrics", {})

        if not isinstance(metrics, Mapping) or not metrics:
            if provider_status not in {
                "baseline_only",
                "candidate_only",
                "model_changed",
                "model_unknown",
                "not_comparable",
            }:
                print("  Nenhuma métrica disponível para comparação.")

            continue

        for metric_name, metric in metrics.items():
            if not isinstance(metric, Mapping):
                continue

            metric_status = metric.get("status", "not_comparable")

            baseline_value = _format_comparison_value(
                metric.get("baseline"),
                metric_name,
            )

            candidate_value = _format_comparison_value(
                metric.get("candidate"),
                metric_name,
            )

            delta = metric.get("delta")

            delta_pct = metric.get("delta_pct")

            if delta is None:
                difference_text = "variação: N/D"

            elif delta_pct is None:
                difference_text = f"variação: {delta}"

            else:
                difference_text = f"variação: {delta} ({delta_pct:+.2f}%)"

            print(
                f"  {_comparison_metric_label(metric_name)}: "
                f"{baseline_value} -> {candidate_value} | "
                f"{difference_text} | "
                f"{str(metric_status).upper()}"
            )

    print()
    print("=" * 88)
    print("Comparação concluída. Nenhum provedor de IA foi executado.")
    print()


# ==========================================================
# VALIDAÇÃO DOS SQLs DE REFERÊNCIA
# ==========================================================


def validate_reference_cases(
    cases: list,
    executor: SQLExecutor | None = None,
) -> dict[str, Any]:
    """Valida os SQLs de referência sem executar modelos de IA."""

    reference_executor = executor if executor is not None else SQLExecutor()

    results = []

    for index, case in enumerate(cases, start=1):
        start = perf_counter()

        try:
            rows = reference_executor.execute(case.reference_sql)

            if not isinstance(rows, list):
                raise TypeError("O SQLExecutor não retornou uma lista de registros.")

            if not all(isinstance(row, dict) for row in rows):
                raise TypeError("O SQLExecutor retornou registros em formato inválido.")

            columns = set().union(*[set(row.keys()) for row in rows]) if rows else set()

            invalid_column_names = [
                column for column in columns if not isinstance(column, str)
            ]

            if invalid_column_names:
                raise TypeError("O resultado contém nomes de colunas inválidos.")

            missing_columns = set(case.required_columns) - columns if rows else set()

            if missing_columns:
                raise ValueError(
                    "Colunas obrigatórias ausentes: "
                    + ", ".join(sorted(missing_columns))
                )

            duration_ms = round(
                (perf_counter() - start) * 1000,
                2,
            )

            schema_validation = "validated" if rows else "unavailable_empty_result"

            result = {
                "case_id": case.case_id,
                "category": case.category,
                "status": "passed",
                "row_count": len(rows),
                "columns": sorted(columns),
                "required_columns": list(case.required_columns),
                "schema_validation": schema_validation,
                "duration_ms": duration_ms,
                "error": None,
            }

            logger.info(
                "SQL de referência validado",
                extra={
                    "event": "evaluation_reference_validated",
                    "case_id": case.case_id,
                    "row_count": len(rows),
                },
            )

        except Exception as exc:
            duration_ms = round(
                (perf_counter() - start) * 1000,
                2,
            )

            result = {
                "case_id": case.case_id,
                "category": case.category,
                "status": "error",
                "row_count": None,
                "columns": [],
                "required_columns": list(case.required_columns),
                "schema_validation": "failed",
                "duration_ms": duration_ms,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

            logger.exception(
                "Falha na validação do SQL de referência",
                extra={
                    "event": "evaluation_reference_validation_failed",
                    "case_id": case.case_id,
                    "position": index,
                },
            )

        results.append(result)

    passed = sum(result["status"] == "passed" for result in results)

    errors = sum(result["status"] == "error" for result in results)

    return {
        "mode": "reference_validation",
        "database": "DuckDB",
        "summary": {
            "total_cases": len(results),
            "passed_cases": passed,
            "error_cases": errors,
            "pass_rate": round(
                passed / len(results) * 100,
                2,
            )
            if results
            else 0.0,
        },
        "results": results,
    }


def print_validation_report(
    payload: dict[str, Any],
) -> None:
    """Exibe o resultado da validação dos SQLs."""

    summary = payload["summary"]

    print()
    print("AI EVALUATION ENGINE — VALIDAÇÃO DE REFERÊNCIAS")
    print("=" * 64)

    print(f"Cenários: {summary['total_cases']}")

    print(f"Aprovados: {summary['passed_cases']}")

    print(f"Erros: {summary['error_cases']}")

    print(f"Taxa de validação: {summary['pass_rate']:.2f}%")

    print()
    print("-" * 64)

    for result in payload["results"]:
        status = result["status"].upper()

        print(f"[{status}] {result['case_id']}")

        print(f"  Registros: {result['row_count']}")

        print(f"  Duração: {result['duration_ms']:.2f} ms")

        if result["error"]:
            print(f"  Erro: {result['error']}")

        print()


# ==========================================================
# EXECUÇÃO COMPLETA INDIVIDUAL
# ==========================================================


def execute_evaluation(
    cases: list,
    provider: str,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Executa o EvaluationRunner com o provedor selecionado."""

    from ai_analyst.evaluation.runner import EvaluationRunner

    runner = EvaluationRunner(
        provider_name=provider,
        dataset_path=dataset_path,
    )

    report = runner.run(cases=cases)

    return report.to_dict()


def print_evaluation_report(
    payload: dict[str, Any],
) -> None:
    """Exibe um resumo legível da avaliação."""

    summary = payload["summary"]

    print()
    print("AI BUSINESS ANALYST — EVALUATION REPORT")
    print("=" * 72)

    print(f"Run ID: {payload['run_id']}")

    print(f"Provedor: {payload['provider']}")

    print(f"Modelo: {payload['model'] or 'Não informado'}")

    print()

    print(f"Total de cenários: {summary['total_cases']}")

    print(f"Aprovados: {summary['passed_cases']}")

    print(f"Reprovados: {summary['failed_cases']}")

    print(f"Erros: {summary['error_cases']}")

    print(f"Ignorados: {summary['skipped_cases']}")

    print(f"Taxa de aprovação: {summary['pass_rate']:.2f}%")

    print(f"Duração total: {summary['duration_ms']:.2f} ms")

    metrics = payload.get("metadata", {}).get(
        "metrics",
        {},
    )

    metric_summary = metrics.get("summary", {})

    if metric_summary:
        print()

        print(f"Score médio: {metric_summary.get('average_score', 0.0):.4f}")

        print(f"Duração média por análise: {metric_summary.get('average_duration_ms')}")

        print(f"Taxa de cache: {metric_summary.get('cache_hit_rate')}")

        adherence = metric_summary.get("analysis_mode_adherence")

        if isinstance(adherence, Mapping):
            expected = adherence.get("expected_cases", 0)
            observed = adherence.get("observed_cases", 0)
            rate = adherence.get("rate")

            if expected:
                rate_text = f"{rate:.2f}%" if rate is not None else "Indisponível"

                print(
                    f"Aderência ao planejamento: {rate_text} "
                    f"({adherence.get('matches', 0)}/{observed} "
                    f"modos observados; {expected} esperados)"
                )

    print()
    print("-" * 72)

    for result in payload.get("results", []):
        status = result["status"].upper()

        score = result.get("score", 0.0)

        print(f"[{status}] {result['case_id']} | Score: {score:.4f}")

        if result.get("error"):
            print(f"  Erro: {result['error']}")

        for difference in result.get("differences", []):
            print(f"  Diferença: {difference}")

    print()


# ==========================================================
# EXECUÇÃO DO BENCHMARK COMPARATIVO
# ==========================================================


def execute_benchmark(
    cases: list,
    providers: Sequence[str],
    dataset_path: Path | None = None,
    history_store: BenchmarkHistory | None = None,
) -> dict[str, Any]:
    """Executa o BenchmarkRunner com persistência histórica."""

    from ai_analyst.evaluation.benchmark import BenchmarkRunner

    effective_history_store = (
        history_store if history_store is not None else BenchmarkHistory()
    )

    benchmark = BenchmarkRunner(
        providers=providers,
        dataset_path=dataset_path,
        history_store=effective_history_store,
    )

    return benchmark.run(
        cases=cases,
    )


# ==========================================================
# FORMATAÇÃO DO BENCHMARK
# ==========================================================


def _format_percentage(
    value: Any,
) -> str:
    """Formata uma porcentagem ou informa indisponibilidade."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/D"

    return f"{value:.2f}%"


def _format_score(
    value: Any,
) -> str:
    """Formata um score numérico."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/D"

    return f"{value:.4f}"


def _format_duration(
    value: Any,
) -> str:
    """Formata uma duração em milissegundos."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "N/D"

    return f"{value:.2f} ms"


def print_benchmark_report(
    payload: dict[str, Any],
) -> None:
    """Exibe o resumo comparativo do benchmark."""

    dataset = payload.get("dataset", {})

    summary = payload.get("summary", {})

    comparison = payload.get("comparison", {})

    providers = comparison.get(
        "by_provider",
        {},
    )

    print()
    print("AI BUSINESS ANALYST — MULTI-LLM BENCHMARK")
    print("=" * 88)

    print(f"Benchmark ID: {payload.get('benchmark_id', 'N/D')}")

    print(f"Provedores: {summary.get('provider_count', 0)}")

    print(f"Cenários: {dataset.get('case_count', 0)}")

    print(f"Dataset SHA-256: {dataset.get('sha256', 'N/D')}")

    print()

    print(f"Provedores concluídos: {summary.get('completed_providers', 0)}")

    print(f"Provedores com falha: {summary.get('failed_providers', 0)}")

    print()
    print("-" * 88)
    print("COMPARAÇÃO POR PROVEDOR")
    print("-" * 88)

    for provider, execution in providers.items():
        print()
        print(f"PROVEDOR: {provider.upper()}")

        if execution.get("execution_status") != "completed":
            print("  Status: ERRO")

            print(f"  Tipo: {execution.get('error_type') or 'N/D'}")

            print(f"  Mensagem: {execution.get('error') or 'Erro desconhecido'}")

            continue

        metrics = execution.get(
            "metrics",
            {},
        )

        adherence = metrics.get("analysis_mode_adherence")

        print(f"  Modelo: {execution.get('model') or 'Não informado'}")

        print(f"  Taxa de aprovação: {_format_percentage(metrics.get('pass_rate'))}")

        print(f"  Score médio: {_format_score(metrics.get('average_score'))}")

        print(
            f"  Latência média: {_format_duration(metrics.get('average_duration_ms'))}"
        )

        print(f"  Taxa de cache: {_format_percentage(metrics.get('cache_hit_rate'))}")

        if isinstance(adherence, Mapping):
            rate = adherence.get("rate")

            print(f"  Aderência ao planejamento: {_format_percentage(rate)}")

            print(
                f"  Modos observados: "
                f"{adherence.get('observed_cases', 0)}"
                f"/{adherence.get('expected_cases', 0)}"
            )

        print(f"  Erros: {metrics.get('error_cases', 'N/D')}")

        print(f"  Cenários ignorados: {metrics.get('skipped_cases', 'N/D')}")

        print(
            "  Custo: N/D (telemetria de tokens e preços "
            "ainda não integrada ao benchmark)"
        )

    # ------------------------------------------------------
    # COMPARAÇÃO POR CATEGORIA
    # ------------------------------------------------------

    categories = comparison.get(
        "by_category",
        {},
    )

    if categories:
        print()
        print("-" * 88)
        print("COMPARAÇÃO POR CATEGORIA")
        print("-" * 88)

        for category, provider_metrics in categories.items():
            print()
            print(f"CATEGORIA: {category}")

            for provider, metrics in provider_metrics.items():
                print(
                    f"  {provider}: "
                    f"aprovação={_format_percentage(metrics.get('pass_rate'))} | "
                    f"score={_format_score(metrics.get('average_score'))}"
                )

    # ------------------------------------------------------
    # COMPARAÇÃO POR CENÁRIO
    # ------------------------------------------------------

    cases = comparison.get(
        "by_case",
        {},
    )

    if cases:
        print()
        print("-" * 88)
        print("COMPARAÇÃO POR CENÁRIO")
        print("-" * 88)

        for case_id, case_data in cases.items():
            expected_mode = case_data.get("expected_analysis_mode")

            print()
            print(f"CENÁRIO: {case_id}")

            if expected_mode is not None:
                print(f"  Modo esperado: {expected_mode}")

            for provider, result in case_data.get(
                "providers",
                {},
            ).items():
                mode = result.get("analysis_mode") or "N/D"

                print(
                    f"  {provider}: "
                    f"status={result.get('status', 'N/D')} | "
                    f"score={_format_score(result.get('score'))} | "
                    f"modo={mode}"
                )

    print()
    print("=" * 88)
    print(
        "Benchmark finalizado. Nenhum ranking ponderado "
        "ou custo não verificado foi inferido."
    )
    print()


# ==========================================================
# CÓDIGOS DE SAÍDA
# ==========================================================


def evaluation_exit_code(
    payload: dict[str, Any],
) -> int:
    """Determina se uma avaliação completa foi aprovada."""

    summary = payload["summary"]

    incomplete = (
        summary["failed_cases"] > 0
        or summary["error_cases"] > 0
        or summary["skipped_cases"] > 0
    )

    return EXIT_FAILURE if incomplete else EXIT_SUCCESS


def validation_exit_code(
    payload: dict[str, Any],
) -> int:
    """Determina se todos os SQLs de referência foram validados."""

    return EXIT_FAILURE if payload["summary"]["error_cases"] > 0 else EXIT_SUCCESS


def benchmark_exit_code(
    payload: dict[str, Any],
) -> int:
    """Valida a integridade e o resultado completo do benchmark."""

    if not isinstance(payload, Mapping):
        return EXIT_FAILURE

    benchmark_summary = payload.get("summary", {})

    dataset = payload.get("dataset", {})

    configuration = payload.get("configuration", {})

    provider_runs = payload.get("provider_runs", {})

    comparison = payload.get("comparison", {})

    if not all(
        isinstance(value, Mapping)
        for value in (
            benchmark_summary,
            dataset,
            configuration,
            provider_runs,
            comparison,
        )
    ):
        return EXIT_FAILURE

    expected_case_ids = dataset.get("case_ids")

    expected_case_count = dataset.get("case_count")

    if (
        not isinstance(expected_case_ids, list)
        or not expected_case_ids
        or not all(
            isinstance(case_id, str) and case_id.strip()
            for case_id in expected_case_ids
        )
    ):
        return EXIT_FAILURE

    expected_case_set = set(expected_case_ids)

    if len(expected_case_set) != len(expected_case_ids):
        return EXIT_FAILURE

    if (
        isinstance(expected_case_count, bool)
        or not isinstance(expected_case_count, int)
        or expected_case_count != len(expected_case_ids)
    ):
        return EXIT_FAILURE

    expected_providers = configuration.get("providers")

    if (
        not isinstance(expected_providers, list)
        or len(expected_providers) < 2
        or not all(
            isinstance(provider, str) and provider.strip()
            for provider in expected_providers
        )
    ):
        return EXIT_FAILURE

    expected_provider_set = set(expected_providers)

    if len(expected_provider_set) != len(expected_providers):
        return EXIT_FAILURE

    if set(provider_runs) != expected_provider_set:
        return EXIT_FAILURE

    if benchmark_summary.get("provider_count") != len(expected_providers):
        return EXIT_FAILURE

    if configuration.get("provider_count") != len(expected_providers):
        return EXIT_FAILURE

    if configuration.get("scenario_count") != expected_case_count:
        return EXIT_FAILURE

    if configuration.get("same_dataset_for_all_providers") is not True:
        return EXIT_FAILURE

    if benchmark_summary.get("failed_providers", 0) > 0:
        return EXIT_FAILURE

    if benchmark_summary.get("completed_providers") != len(expected_providers):
        return EXIT_FAILURE

    for provider, execution in provider_runs.items():
        if not isinstance(execution, Mapping):
            return EXIT_FAILURE

        if execution.get("execution_status") != "completed":
            return EXIT_FAILURE

        report = execution.get("report")

        if not isinstance(report, Mapping):
            return EXIT_FAILURE

        report_summary = report.get("summary")

        if not isinstance(report_summary, Mapping):
            return EXIT_FAILURE

        if report_summary.get("total_cases") != expected_case_count:
            return EXIT_FAILURE

        if (
            report_summary.get("failed_cases", 0) > 0
            or report_summary.get("error_cases", 0) > 0
            or report_summary.get("skipped_cases", 0) > 0
        ):
            return EXIT_FAILURE

        report_results = report.get("results")

        if not isinstance(report_results, list):
            return EXIT_FAILURE

        report_case_ids = []

        for result in report_results:
            if not isinstance(result, Mapping):
                return EXIT_FAILURE

            case_id = result.get("case_id")

            if not isinstance(case_id, str) or not case_id.strip():
                return EXIT_FAILURE

            report_case_ids.append(case_id)

            if result.get("status") != "passed":
                return EXIT_FAILURE

        if len(report_case_ids) != expected_case_count:
            return EXIT_FAILURE

        if len(set(report_case_ids)) != expected_case_count:
            return EXIT_FAILURE

        if set(report_case_ids) != expected_case_set:
            return EXIT_FAILURE

    provider_comparison = comparison.get("by_provider")

    if not isinstance(provider_comparison, Mapping):
        return EXIT_FAILURE

    if set(provider_comparison) != expected_provider_set:
        return EXIT_FAILURE

    for provider, provider_data in provider_comparison.items():
        if not isinstance(provider_data, Mapping):
            return EXIT_FAILURE

        if provider_data.get("execution_status") != "completed":
            return EXIT_FAILURE

        if not isinstance(
            provider_data.get("metrics"),
            Mapping,
        ):
            return EXIT_FAILURE

    case_comparison = comparison.get("by_case")

    if not isinstance(case_comparison, Mapping):
        return EXIT_FAILURE

    if set(case_comparison) != expected_case_set:
        return EXIT_FAILURE

    for case_id, case_data in case_comparison.items():
        if not isinstance(case_data, Mapping):
            return EXIT_FAILURE

        provider_results = case_data.get("providers")

        if not isinstance(provider_results, Mapping):
            return EXIT_FAILURE

        if set(provider_results) != expected_provider_set:
            return EXIT_FAILURE

        for provider, result in provider_results.items():
            if not isinstance(result, Mapping):
                return EXIT_FAILURE

            if result.get("status") != "passed":
                return EXIT_FAILURE

            score = result.get("score")

            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0.0 <= score <= 1.0
            ):
                return EXIT_FAILURE

    return EXIT_SUCCESS


# ==========================================================
# PONTO DE ENTRADA
# ==========================================================


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Executa a interface de linha de comando."""

    parser = build_parser()

    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    # ------------------------------------------------------
    # 1. Identificar os modos sem chamadas de IA.
    # ------------------------------------------------------

    is_comparison_mode = args.compare_benchmarks is not None

    is_history_mode = (
        args.list_benchmarks or args.show_benchmark is not None or is_comparison_mode
    )

    is_non_llm_mode = args.list_cases or args.validate_only or is_history_mode

    # ------------------------------------------------------
    # 2. Validar autorização explícita de chamadas de IA.
    # ------------------------------------------------------

    if args.allow_llm and is_non_llm_mode:
        parser.error(
            "--allow-llm não deve ser utilizado com modos "
            "que não executam provedores de IA."
        )

    if not is_non_llm_mode and not args.allow_llm:
        parser.error(
            "A execução completa pode consumir quotas e custos "
            "de provedores externos. Para autorizar chamadas reais, "
            "informe explicitamente --allow-llm. "
            "Utilize --list-cases, --validate-only, "
            "--list-benchmarks, --show-benchmark ou "
            "--compare-benchmarks para operações sem chamadas de IA."
        )

    if args.list_cases and args.output:
        parser.error("--output não é utilizado com --list-cases.")

    # ------------------------------------------------------
    # 3. Validar configuração do benchmark.
    # ------------------------------------------------------

    if args.benchmark is not None:
        if len(args.benchmark) < 2:
            parser.error("--benchmark exige pelo menos dois provedores.")

        if len(args.benchmark) != len(set(args.benchmark)):
            parser.error("--benchmark não aceita provedores duplicados.")

    # ------------------------------------------------------
    # 4. Consultar o histórico de benchmarks.
    #
    # Estas operações são independentes do dataset.
    # ------------------------------------------------------

    if is_history_mode:
        try:
            history_store = BenchmarkHistory()

        except Exception as exc:
            logger.exception("Falha ao inicializar o histórico de benchmarks")

            print(
                f"Erro ao inicializar o histórico: {exc}",
                file=sys.stderr,
            )

            return EXIT_ERROR

        # --------------------------------------------------
        # 4.1. Comparação entre duas execuções históricas.
        # --------------------------------------------------

        if is_comparison_mode:
            baseline_id, candidate_id = args.compare_benchmarks

            try:
                payload = execute_benchmark_comparison(
                    history_store=history_store,
                    baseline_id=baseline_id,
                    candidate_id=candidate_id,
                )

            except ValueError as exc:
                logger.warning(
                    "Não foi possível comparar os benchmarks: %s",
                    exc,
                )

                print(
                    f"Erro na comparação: {exc}",
                    file=sys.stderr,
                )

                return EXIT_FAILURE

            except Exception as exc:
                logger.exception("Falha inesperada na comparação de benchmarks")

                print(
                    f"Erro durante a comparação: {exc}",
                    file=sys.stderr,
                )

                return EXIT_ERROR

            print_benchmark_comparison_report(payload)

            if args.output:
                try:
                    saved_path = write_json_report(
                        args.output,
                        payload,
                    )

                    print(f"Relatório de comparação salvo em: {saved_path.resolve()}")

                except Exception as exc:
                    logger.exception("Falha ao salvar relatório de comparação")

                    print(
                        f"Erro ao salvar relatório de comparação: {exc}",
                        file=sys.stderr,
                    )

                    return EXIT_ERROR

            return benchmark_comparison_exit_code(payload)

        # --------------------------------------------------
        # 4.2. Listagem paginada do histórico.
        # --------------------------------------------------

        if args.list_benchmarks:
            try:
                history_payload = list_benchmark_history(
                    history_store=history_store,
                    limit=args.limit,
                    offset=args.offset,
                    provider=args.filter_provider,
                    model=args.filter_model,
                )

            except Exception as exc:
                logger.exception("Falha ao consultar o histórico de benchmarks")

                print(
                    f"Erro ao consultar o histórico: {exc}",
                    file=sys.stderr,
                )

                return EXIT_ERROR

            pagination = history_payload["pagination"]

            print_benchmark_history(
                entries=history_payload["items"],
                total=pagination["total"],
                limit=pagination["limit"],
                offset=pagination["offset"],
                provider=args.filter_provider,
                model=args.filter_model,
            )

            if args.output:
                try:
                    saved_path = write_json_report(
                        args.output,
                        history_payload,
                    )

                    print(f"Histórico exportado em: {saved_path.resolve()}")

                except Exception as exc:
                    logger.exception("Falha ao exportar o histórico")

                    print(
                        f"Erro ao exportar o histórico: {exc}",
                        file=sys.stderr,
                    )

                    return EXIT_ERROR

            return EXIT_SUCCESS

        # --------------------------------------------------
        # 4.3. Recuperação do relatório completo pelo ID.
        # --------------------------------------------------

        if args.show_benchmark is not None:
            try:
                payload = show_benchmark_history(
                    history_store=history_store,
                    benchmark_id=args.show_benchmark,
                )

            except Exception as exc:
                logger.exception("Falha ao recuperar o benchmark do histórico")

                print(
                    f"Erro ao recuperar o benchmark: {exc}",
                    file=sys.stderr,
                )

                return EXIT_ERROR

            if payload is None:
                print(
                    f"Benchmark não encontrado: {args.show_benchmark}",
                    file=sys.stderr,
                )

                return EXIT_FAILURE

            print_benchmark_report(payload)

            if args.output:
                try:
                    saved_path = write_json_report(
                        args.output,
                        payload,
                    )

                    print(f"Relatório histórico exportado em: {saved_path.resolve()}")

                except Exception as exc:
                    logger.exception("Falha ao exportar o relatório histórico")

                    print(
                        f"Erro ao exportar o relatório: {exc}",
                        file=sys.stderr,
                    )

                    return EXIT_ERROR

            return EXIT_SUCCESS

    # ------------------------------------------------------
    # 5. Carregar e filtrar cenários.
    # ------------------------------------------------------

    try:
        cases = load_cases(args.dataset)

        selected_cases = select_cases(
            cases,
            args.case_ids,
        )

    except Exception as exc:
        print(
            f"Erro ao carregar dataset: {exc}",
            file=sys.stderr,
        )

        logger.exception("Falha ao preparar os cenários de avaliação")

        return EXIT_ERROR

    # ------------------------------------------------------
    # 6. Listagem sem execução.
    # ------------------------------------------------------

    if args.list_cases:
        list_cases(selected_cases)
        return EXIT_SUCCESS

    # ------------------------------------------------------
    # 7. Validação de SQL sem IA.
    # ------------------------------------------------------

    if args.validate_only:
        payload = validate_reference_cases(selected_cases)

        print_validation_report(payload)

        if args.output:
            try:
                saved_path = write_json_report(
                    args.output,
                    payload,
                )

                print(f"Relatório salvo em: {saved_path.resolve()}")

            except Exception as exc:
                print(
                    f"Falha ao salvar relatório: {exc}",
                    file=sys.stderr,
                )

                return EXIT_ERROR

        return validation_exit_code(payload)

    # ------------------------------------------------------
    # 8. Benchmark comparativo com autorização explícita.
    # ------------------------------------------------------

    if args.benchmark is not None:
        try:
            payload = execute_benchmark(
                cases=selected_cases,
                providers=args.benchmark,
                dataset_path=args.dataset,
            )

        except Exception as exc:
            logger.exception("Falha na execução do benchmark comparativo")

            print(
                f"Erro durante o benchmark: {exc}",
                file=sys.stderr,
            )

            return EXIT_ERROR

        print_benchmark_report(payload)

        if args.output:
            try:
                saved_path = write_json_report(
                    args.output,
                    payload,
                )

                print(f"Relatório de benchmark salvo em: {saved_path.resolve()}")

            except Exception as exc:
                logger.exception("Falha ao salvar relatório do benchmark")

                print(
                    f"Erro ao salvar relatório: {exc}",
                    file=sys.stderr,
                )

                return EXIT_ERROR

        return benchmark_exit_code(payload)

    # ------------------------------------------------------
    # 9. Execução individual com autorização explícita.
    # ------------------------------------------------------

    try:
        payload = execute_evaluation(
            cases=selected_cases,
            provider=args.provider,
            dataset_path=args.dataset,
        )

    except Exception as exc:
        logger.exception("Falha na execução completa do Evaluation Engine")

        print(
            f"Erro durante a avaliação: {exc}",
            file=sys.stderr,
        )

        return EXIT_ERROR

    print_evaluation_report(payload)

    # ------------------------------------------------------
    # 10. Exportação JSON.
    # ------------------------------------------------------

    if args.output:
        try:
            saved_path = write_json_report(
                args.output,
                payload,
            )

            print(f"Relatório JSON salvo em: {saved_path.resolve()}")

        except Exception as exc:
            logger.exception("Falha ao salvar relatório JSON")

            print(
                f"Erro ao salvar relatório: {exc}",
                file=sys.stderr,
            )

            return EXIT_ERROR

    # ------------------------------------------------------
    # 11. Código de saída.
    # ------------------------------------------------------

    return evaluation_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
