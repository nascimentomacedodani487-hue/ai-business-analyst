"""Testes de integração do benchmark com a CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_analyst.evaluation import benchmark as benchmark_module
from ai_analyst.evaluation import cli
from ai_analyst.evaluation.models import EvaluationCase

# ==========================================================
# FÁBRICAS DE DADOS
# ==========================================================


def make_case(
    case_id: str = "case-1",
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"Pergunta de teste {case_id}",
        reference_sql="SELECT 1",
        category="volume",
        comparison_mode="exact",
    )


def make_benchmark_payload(
    case_ids: tuple[str, ...] = ("case-1", "case-2"),
) -> dict[str, Any]:
    """Monta um benchmark fictício com dois provedores."""

    providers = ("gemini", "openai")

    provider_runs = {}

    by_provider = {}

    for provider in providers:
        report_results = [
            {
                "case_id": case_id,
                "question": f"Pergunta {case_id}",
                "status": "passed",
                "score": 1.0,
                "expected_rows": [{"total_orders": 60000}],
                "actual_rows": [{"total_orders": 60000}],
                "differences": [],
                "analysis_id": f"analysis-{provider}-{index}",
                "provider": provider,
                "model": "fake-model",
                "duration_ms": 100.0,
                "cache_hit": False,
                "error": None,
                "metadata": {
                    "category": "volume",
                    "analysis_mode": "single_query",
                },
            }
            for index, case_id in enumerate(case_ids)
        ]

        provider_runs[provider] = {
            "execution_status": "completed",
            "report": {
                "run_id": f"benchmark-test-{provider}",
                "provider": provider,
                "model": "fake-model",
                "summary": {
                    "total_cases": len(case_ids),
                    "evaluated_cases": len(case_ids),
                    "passed_cases": len(case_ids),
                    "failed_cases": 0,
                    "error_cases": 0,
                    "skipped_cases": 0,
                    "pass_rate": 100.0,
                    "duration_ms": 1000.0,
                },
                "results": report_results,
            },
            "error": None,
            "error_type": None,
        }

        by_provider[provider] = {
            "execution_status": "completed",
            "model": "fake-model",
            "run_id": f"benchmark-test-{provider}",
            "metrics": {
                "total_cases": len(case_ids),
                "evaluated_cases": len(case_ids),
                "passed_cases": len(case_ids),
                "failed_cases": 0,
                "error_cases": 0,
                "skipped_cases": 0,
                "pass_rate": 100.0,
                "average_score": 1.0,
                "average_duration_ms": 100.0,
                "cache_hit_rate": 0.0,
                "analysis_mode_adherence": None,
            },
        }

    by_case = {
        case_id: {
            "category": "volume",
            "question": f"Pergunta {case_id}",
            "expected_analysis_mode": None,
            "providers": {
                provider: {
                    "status": "passed",
                    "score": 1.0,
                    "duration_ms": 100.0,
                    "analysis_mode": "single_query",
                    "analysis_mode_matches": None,
                    "cache_hit": False,
                    "error": None,
                }
                for provider in providers
            },
        }
        for case_id in case_ids
    }

    return {
        "schema_version": 1,
        "benchmark_id": "benchmark-test-001",
        "created_at": "2026-11-09T12:00:00+00:00",
        "dataset": {
            "version": 1,
            "source": None,
            "case_count": len(case_ids),
            "case_ids": list(case_ids),
            "sha256": "a" * 64,
        },
        "configuration": {
            "providers": list(providers),
            "provider_count": len(providers),
            "scenario_count": len(case_ids),
            "execution_strategy": "sequential",
            "same_dataset_for_all_providers": True,
        },
        "summary": {
            "provider_count": len(providers),
            "completed_providers": len(providers),
            "failed_providers": 0,
            "all_providers_completed": True,
        },
        "provider_runs": provider_runs,
        "comparison": {
            "by_provider": by_provider,
            "by_category": {},
            "by_case": by_case,
        },
    }


def make_benchmark_history_entry(
    benchmark_id: str = "benchmark-test-001",
) -> dict[str, Any]:
    """Monta um registro resumido para a listagem do histórico."""

    return {
        "benchmark_id": benchmark_id,
        "created_at": "2026-11-09T12:00:00+00:00",
        "dataset_case_count": 2,
        "dataset_sha256": "a" * 64,
        "providers": ["gemini", "openai"],
        "models": ["fake-model"],
        "provider_count": 2,
        "completed_providers": 2,
        "failed_providers": 0,
    }


# ==========================================================
# ARGUMENTOS DO BENCHMARK
# ==========================================================


def test_parser_accepts_benchmark_providers():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--benchmark",
            "gemini",
            "openai",
            "anthropic",
            "--allow-llm",
        ]
    )

    assert args.benchmark == [
        "gemini",
        "openai",
        "anthropic",
    ]

    assert args.allow_llm is True


def test_parser_defaults_benchmark_to_none():
    parser = cli.build_parser()

    args = parser.parse_args([])

    assert args.benchmark is None


def test_parser_rejects_benchmark_with_validate_only():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--benchmark",
                "gemini",
                "openai",
                "--validate-only",
            ]
        )

    assert exc.value.code == 2


# ==========================================================
# ARGUMENTOS DO HISTÓRICO
# ==========================================================


def test_parser_accepts_list_benchmarks():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--list-benchmarks",
        ]
    )

    assert args.list_benchmarks is True
    assert args.show_benchmark is None


def test_parser_accepts_show_benchmark():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--show-benchmark",
            "benchmark-test-001",
        ]
    )

    assert args.show_benchmark == "benchmark-test-001"
    assert args.list_benchmarks is False


def test_parser_accepts_benchmark_history_filters():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--list-benchmarks",
            "--filter-provider",
            "gemini",
            "--filter-model",
            "gemini-2.5-flash",
            "--limit",
            "5",
            "--offset",
            "10",
        ]
    )

    assert args.list_benchmarks is True
    assert args.filter_provider == "gemini"
    assert args.filter_model == "gemini-2.5-flash"
    assert args.limit == 5
    assert args.offset == 10


def test_parser_defaults_benchmark_history_pagination():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--list-benchmarks",
        ]
    )

    assert args.limit == 20
    assert args.offset == 0


def test_parser_rejects_list_benchmarks_with_benchmark_execution():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--list-benchmarks",
                "--benchmark",
                "gemini",
                "openai",
            ]
        )

    assert exc.value.code == 2


def test_parser_rejects_show_benchmark_with_benchmark_execution():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--show-benchmark",
                "benchmark-test-001",
                "--benchmark",
                "gemini",
                "openai",
            ]
        )

    assert exc.value.code == 2


def test_parser_rejects_list_and_show_benchmark_together():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--list-benchmarks",
                "--show-benchmark",
                "benchmark-test-001",
            ]
        )

    assert exc.value.code == 2


# ==========================================================
# EXECUÇÃO DO BENCHMARK
# ==========================================================


def test_execute_benchmark_uses_benchmark_runner(
    monkeypatch,
):
    cases = [
        make_case(),
    ]

    expected_payload = make_benchmark_payload(
        case_ids=("case-1",),
    )

    captured = {}

    class FakeHistoryStore:
        """Simula o armazenamento padrão da CLI."""

    class FakeBenchmarkRunner:
        def __init__(
            self,
            providers,
            dataset_path=None,
            history_store=None,
        ):
            captured["providers"] = providers
            captured["dataset_path"] = dataset_path
            captured["history_store"] = history_store

        def run(self, cases):
            captured["cases"] = cases

            return expected_payload

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    monkeypatch.setattr(
        benchmark_module,
        "BenchmarkRunner",
        FakeBenchmarkRunner,
    )

    dataset_path = Path("custom_dataset.json")

    payload = cli.execute_benchmark(
        cases=cases,
        providers=["gemini", "openai"],
        dataset_path=dataset_path,
    )

    assert payload == expected_payload

    assert captured["providers"] == [
        "gemini",
        "openai",
    ]

    assert captured["dataset_path"] == dataset_path

    assert captured["cases"] == cases

    assert isinstance(
        captured["history_store"],
        FakeHistoryStore,
    )


def test_execute_benchmark_accepts_injected_history_store(
    monkeypatch,
):
    """Um histórico externo deve ser reutilizado sem criar outro."""

    cases = [
        make_case(),
    ]

    expected_payload = make_benchmark_payload(
        case_ids=("case-1",),
    )

    history_store = object()

    captured = {}

    class FakeBenchmarkRunner:
        def __init__(
            self,
            providers,
            dataset_path=None,
            history_store=None,
        ):
            captured["providers"] = providers
            captured["dataset_path"] = dataset_path
            captured["history_store"] = history_store

        def run(self, cases):
            captured["cases"] = cases

            return expected_payload

    def unexpected_history_creation():
        pytest.fail(
            "A CLI não deve criar outro histórico quando "
            "um armazenamento já foi injetado."
        )

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        unexpected_history_creation,
        raising=False,
    )

    monkeypatch.setattr(
        benchmark_module,
        "BenchmarkRunner",
        FakeBenchmarkRunner,
    )

    payload = cli.execute_benchmark(
        cases=cases,
        providers=["gemini", "openai"],
        dataset_path=None,
        history_store=history_store,
    )

    assert payload == expected_payload

    assert captured["history_store"] is history_store
    assert captured["cases"] == cases


# ==========================================================
# CONSULTA DO HISTÓRICO
# ==========================================================


def test_main_lists_benchmarks_without_llm_authorization(
    monkeypatch,
    capsys,
):
    """A listagem do histórico não deve exigir autorização de LLM."""

    entry = make_benchmark_history_entry()

    captured = {}

    class FakeHistoryStore:
        def __init__(self):
            captured["initialized"] = True

        def list_benchmarks(
            self,
            limit,
            offset,
            provider=None,
            model=None,
        ):
            captured["list_arguments"] = {
                "limit": limit,
                "offset": offset,
                "provider": provider,
                "model": model,
            }

            return [entry]

        def count_benchmarks(
            self,
            provider=None,
            model=None,
        ):
            captured["count_arguments"] = {
                "provider": provider,
                "model": model,
            }

            return 1

    def unexpected_benchmark_execution(**kwargs):
        pytest.fail("A listagem do histórico não deve executar benchmarks.")

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        unexpected_benchmark_execution,
    )

    exit_code = cli.main(
        [
            "--list-benchmarks",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert captured["initialized"] is True

    assert captured["list_arguments"] == {
        "limit": 20,
        "offset": 0,
        "provider": None,
        "model": None,
    }

    assert captured["count_arguments"] == {
        "provider": None,
        "model": None,
    }

    assert "benchmark-test-001" in captured_output.out


def test_main_lists_benchmarks_with_provider_model_filters_and_pagination(
    monkeypatch,
    capsys,
):
    """Os filtros devem ser encaminhados ao armazenamento do histórico."""

    entries = [
        make_benchmark_history_entry(
            benchmark_id="benchmark-filtered-001",
        ),
    ]

    captured = {}

    class FakeHistoryStore:
        def list_benchmarks(
            self,
            limit,
            offset,
            provider=None,
            model=None,
        ):
            captured["list_arguments"] = {
                "limit": limit,
                "offset": offset,
                "provider": provider,
                "model": model,
            }

            return entries

        def count_benchmarks(
            self,
            provider=None,
            model=None,
        ):
            captured["count_arguments"] = {
                "provider": provider,
                "model": model,
            }

            return 15

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--list-benchmarks",
            "--filter-provider",
            "gemini",
            "--filter-model",
            "gemini-2.5-flash",
            "--limit",
            "5",
            "--offset",
            "10",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS

    assert captured["list_arguments"] == {
        "limit": 5,
        "offset": 10,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    }

    assert captured["count_arguments"] == {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    }

    assert "benchmark-filtered-001" in captured_output.out


def test_main_lists_benchmarks_handles_empty_history(
    monkeypatch,
    capsys,
):
    class FakeHistoryStore:
        def list_benchmarks(
            self,
            limit,
            offset,
            provider=None,
            model=None,
        ):
            return []

        def count_benchmarks(
            self,
            provider=None,
            model=None,
        ):
            return 0

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--list-benchmarks",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert (
        "Nenhum benchmark encontrado" in captured_output.out
        or "Nenhum registro encontrado" in captured_output.out
        or "Histórico vazio" in captured_output.out
    )


def test_main_shows_benchmark_by_id_without_llm_authorization(
    monkeypatch,
):
    """A consulta por ID deve recuperar o relatório sem executar LLMs."""

    expected_payload = make_benchmark_payload()

    captured = {}

    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            captured["benchmark_id"] = benchmark_id

            return expected_payload

    def fake_print_benchmark_report(payload):
        captured["printed_payload"] = payload

    def unexpected_benchmark_execution(**kwargs):
        pytest.fail("A consulta de um benchmark não deve executar novos modelos.")

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "print_benchmark_report",
        fake_print_benchmark_report,
    )

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        unexpected_benchmark_execution,
    )

    exit_code = cli.main(
        [
            "--show-benchmark",
            "benchmark-test-001",
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert captured["benchmark_id"] == "benchmark-test-001"
    assert captured["printed_payload"] == expected_payload


def test_main_show_benchmark_returns_failure_when_id_does_not_exist(
    monkeypatch,
    capsys,
):
    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return None

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--show-benchmark",
            "benchmark-does-not-exist",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == cli.EXIT_FAILURE
    assert (
        "não encontrado" in captured_output.err.lower()
        or "nao encontrado" in captured_output.err.lower()
    )


# ==========================================================
# FORMATAÇÃO DO RELATÓRIO
# ==========================================================


def test_print_benchmark_report_displays_provider_metrics(
    capsys,
):
    payload = make_benchmark_payload()

    cli.print_benchmark_report(payload)

    captured = capsys.readouterr()

    assert "MULTI-LLM BENCHMARK" in captured.out
    assert "benchmark-test-001" in captured.out
    assert "GEMINI" in captured.out
    assert "OPENAI" in captured.out
    assert "Taxa de aprovação: 100.00%" in captured.out
    assert "Score médio: 1.0000" in captured.out
    assert "Dataset SHA-256" in captured.out


def test_print_benchmark_report_handles_failed_provider(
    capsys,
):
    payload = make_benchmark_payload()

    execution = payload["provider_runs"]["openai"]

    execution["execution_status"] = "error"
    execution["report"] = None
    execution["error"] = "Falha simulada"
    execution["error_type"] = "RuntimeError"

    payload["comparison"]["by_provider"]["openai"] = {
        "execution_status": "error",
        "error": "Falha simulada",
        "error_type": "RuntimeError",
        "metrics": None,
    }

    payload["summary"]["completed_providers"] = 1
    payload["summary"]["failed_providers"] = 1
    payload["summary"]["all_providers_completed"] = False

    cli.print_benchmark_report(payload)

    captured = capsys.readouterr()

    assert "GEMINI" in captured.out
    assert "OPENAI" in captured.out
    assert "Falha simulada" in captured.out
    assert "Status: ERRO" in captured.out


# ==========================================================
# CÓDIGOS DE SAÍDA
# ==========================================================


def test_benchmark_exit_code_returns_success():
    payload = make_benchmark_payload()

    assert cli.benchmark_exit_code(payload) == cli.EXIT_SUCCESS


def test_benchmark_exit_code_returns_failure_when_provider_fails():
    payload = make_benchmark_payload()

    payload["summary"]["failed_providers"] = 1
    payload["summary"]["completed_providers"] = 1
    payload["summary"]["all_providers_completed"] = False

    assert cli.benchmark_exit_code(payload) == cli.EXIT_FAILURE


@pytest.mark.parametrize(
    "status_field",
    [
        "failed_cases",
        "error_cases",
        "skipped_cases",
    ],
)
def test_benchmark_exit_code_returns_failure_for_incomplete_scenarios(
    status_field,
):
    payload = make_benchmark_payload()

    payload["provider_runs"]["gemini"]["report"]["summary"][status_field] = 1

    assert cli.benchmark_exit_code(payload) == cli.EXIT_FAILURE


def test_benchmark_exit_code_rejects_incomplete_case_coverage():
    payload = make_benchmark_payload()

    payload["comparison"]["by_case"].pop("case-2")

    assert cli.benchmark_exit_code(payload) == cli.EXIT_FAILURE


def test_benchmark_exit_code_rejects_missing_provider_case_result():
    payload = make_benchmark_payload()

    payload["comparison"]["by_case"]["case-1"]["providers"].pop("openai")

    assert cli.benchmark_exit_code(payload) == cli.EXIT_FAILURE


def test_benchmark_exit_code_rejects_missing_provider_report():
    payload = make_benchmark_payload()

    payload["provider_runs"]["openai"]["report"] = None

    assert cli.benchmark_exit_code(payload) == cli.EXIT_FAILURE


# ==========================================================
# PONTO DE ENTRADA DA CLI
# ==========================================================


def test_main_benchmark_requires_explicit_llm_authorization():
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--benchmark",
                "gemini",
                "openai",
            ]
        )

    assert exc.value.code == 2


def test_main_benchmark_rejects_single_provider():
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--benchmark",
                "gemini",
                "--allow-llm",
            ]
        )

    assert exc.value.code == 2


def test_main_benchmark_rejects_duplicate_providers():
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--benchmark",
                "gemini",
                "gemini",
                "--allow-llm",
            ]
        )

    assert exc.value.code == 2


def test_main_benchmark_executes_with_explicit_authorization(
    monkeypatch,
):
    cases = [
        make_case("case-1"),
        make_case("case-2"),
    ]

    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    expected_payload = make_benchmark_payload()

    captured = {}

    def fake_execute_benchmark(
        cases,
        providers,
        dataset_path=None,
    ):
        captured["cases"] = cases
        captured["providers"] = providers
        captured["dataset_path"] = dataset_path

        return expected_payload

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        fake_execute_benchmark,
    )

    monkeypatch.setattr(
        cli,
        "print_benchmark_report",
        lambda payload: None,
    )

    exit_code = cli.main(
        [
            "--benchmark",
            "gemini",
            "openai",
            "--allow-llm",
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS

    assert captured["providers"] == [
        "gemini",
        "openai",
    ]

    assert captured["cases"] == cases


def test_main_benchmark_preserves_case_filter(
    monkeypatch,
):
    cases = [
        make_case("case-1"),
        make_case("case-2"),
    ]

    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    expected_payload = make_benchmark_payload(
        case_ids=("case-2",),
    )

    captured = {}

    def fake_execute_benchmark(
        cases,
        providers,
        dataset_path=None,
    ):
        captured["cases"] = cases
        captured["providers"] = providers

        return expected_payload

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        fake_execute_benchmark,
    )

    monkeypatch.setattr(
        cli,
        "print_benchmark_report",
        lambda payload: None,
    )

    exit_code = cli.main(
        [
            "--benchmark",
            "gemini",
            "openai",
            "--allow-llm",
            "--case",
            "case-2",
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS

    assert [case.case_id for case in captured["cases"]] == [
        "case-2",
    ]


def test_main_benchmark_exports_json_report(
    monkeypatch,
    tmp_path,
):
    cases = [
        make_case("case-1"),
        make_case("case-2"),
    ]

    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    expected_payload = make_benchmark_payload()

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        lambda **kwargs: expected_payload,
    )

    monkeypatch.setattr(
        cli,
        "print_benchmark_report",
        lambda payload: None,
    )

    output_path = tmp_path / "reports" / "benchmark.json"

    exit_code = cli.main(
        [
            "--benchmark",
            "gemini",
            "openai",
            "--allow-llm",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS

    assert output_path.exists()

    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_payload["benchmark_id"] == "benchmark-test-001"
    assert saved_payload["dataset"]["case_count"] == 2

    assert set(saved_payload["provider_runs"]) == {
        "gemini",
        "openai",
    }


def test_main_benchmark_returns_failure_for_partial_provider_failure(
    monkeypatch,
):
    cases = [
        make_case("case-1"),
    ]

    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    payload = make_benchmark_payload(
        case_ids=("case-1",),
    )

    payload["summary"]["completed_providers"] = 1
    payload["summary"]["failed_providers"] = 1
    payload["summary"]["all_providers_completed"] = False

    payload["provider_runs"]["openai"]["execution_status"] = "error"
    payload["provider_runs"]["openai"]["report"] = None

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        lambda **kwargs: payload,
    )

    monkeypatch.setattr(
        cli,
        "print_benchmark_report",
        lambda report: None,
    )

    exit_code = cli.main(
        [
            "--benchmark",
            "gemini",
            "openai",
            "--allow-llm",
        ]
    )

    assert exit_code == cli.EXIT_FAILURE


def test_main_benchmark_handles_execution_exception(
    monkeypatch,
):
    cases = [
        make_case("case-1"),
    ]

    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    def fail_benchmark(**kwargs):
        raise RuntimeError("Falha simulada no benchmark.")

    monkeypatch.setattr(
        cli,
        "execute_benchmark",
        fail_benchmark,
    )

    exit_code = cli.main(
        [
            "--benchmark",
            "gemini",
            "openai",
            "--allow-llm",
        ]
    )

    assert exit_code == cli.EXIT_ERROR
