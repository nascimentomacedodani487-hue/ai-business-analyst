"""Interface de linha de comando do AI Evaluation Engine.

Modos disponíveis:
- Listagem dos cenários.
- Validação dos SQLs de referência sem chamadas de IA.
- Execução individual com autorização explícita para utilizar LLMs.
- Benchmark comparativo entre múltiplos provedores.
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
from ai_analyst.evaluation.dataset import load_evaluation_dataset

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
            "--allow-llm --output reports/benchmark.json"
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
        help="Caminho do arquivo JSON que receberá o relatório.",
    )

    parser.add_argument(
        "--allow-llm",
        action="store_true",
        help=(
            "Autoriza explicitamente chamadas reais aos provedores "
            "de IA. Obrigatório para avaliações completas e benchmarks."
        ),
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
        format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
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
) -> dict[str, Any]:
    """Executa o BenchmarkRunner com os provedores selecionados."""

    from ai_analyst.evaluation.benchmark import BenchmarkRunner

    benchmark = BenchmarkRunner(
        providers=providers,
        dataset_path=dataset_path,
    )

    return benchmark.run(
        cases=cases,
    )


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
    """Valida a integridade e o resultado completo do benchmark.

    Retorna EXIT_FAILURE quando:
    - Existe falha de execução em algum provedor.
    - A configuração de provedores está inconsistente.
    - O dataset não possui uma lista válida de cenários.
    - A comparação não contém exatamente todos os cenários.
    - Um relatório individual não contém todos os cenários.
    - Um cenário não contém resultados de todos os provedores.
    - Existem resultados duplicados ou excedentes.
    - Algum cenário não foi aprovado.

    Retorna EXIT_SUCCESS somente quando a cobertura está completa
    e todos os cenários foram aprovados em todos os provedores.
    """

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

    # ------------------------------------------------------
    # 1. Validar os cenários esperados.
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # 2. Validar os provedores esperados.
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # 3. Validar os relatórios individuais.
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # 4. Validar a comparação por provedor.
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # 5. Validar cobertura exata da comparação por cenário.
    # ------------------------------------------------------

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

        # Cada cenário precisa possuir exatamente um resultado
        # por provedor esperado.
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
    # 1. Validar a autorização de chamadas de IA.
    # ------------------------------------------------------

    if args.allow_llm and (args.list_cases or args.validate_only):
        parser.error(
            "--allow-llm não deve ser usado com --list-cases ou --validate-only."
        )

    if not (args.list_cases or args.validate_only) and not args.allow_llm:
        parser.error(
            "A execução completa pode consumir quotas e custos "
            "de provedores externos. Para autorizar chamadas reais, "
            "informe explicitamente --allow-llm. "
            "Utilize --list-cases ou --validate-only para operações "
            "sem chamadas de IA."
        )

    if args.list_cases and args.output:
        parser.error("--output não é utilizado com --list-cases.")

    # ------------------------------------------------------
    # 2. Validar configuração do benchmark.
    # ------------------------------------------------------

    if args.benchmark is not None:
        if len(args.benchmark) < 2:
            parser.error("--benchmark exige pelo menos dois provedores.")

        if len(args.benchmark) != len(set(args.benchmark)):
            parser.error("--benchmark não aceita provedores duplicados.")

    # ------------------------------------------------------
    # 3. Carregar e filtrar cenários.
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
    # 4. Listagem sem execução.
    # ------------------------------------------------------

    if args.list_cases:
        list_cases(selected_cases)
        return EXIT_SUCCESS

    # ------------------------------------------------------
    # 5. Validação de SQL sem IA.
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
    # 6. Benchmark comparativo com autorização explícita.
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
    # 7. Execução individual com autorização explícita.
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
    # 8. Exportação JSON.
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
    # 9. Código de saída.
    # ------------------------------------------------------

    return evaluation_exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
