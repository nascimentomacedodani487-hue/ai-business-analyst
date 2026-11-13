"""Testes de integração da comparação histórica pela CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_analyst.evaluation import cli

# ==========================================================
# FÁBRICAS DE DADOS
# ==========================================================


def make_provider_metrics(
    pass_rate: float = 100.0,
) -> dict[str, Any]:
    """Monta métricas consolidadas fictícias."""

    return {
        "total_cases": 2,
        "evaluated_cases": 2,
        "passed_cases": 2,
        "failed_cases": 0,
        "error_cases": 0,
        "skipped_cases": 0,
        "pass_rate": pass_rate,
        "scored_cases": 2,
        "average_score": 1.0,
        "average_duration_ms": 100.0,
        "row_accuracy": 100.0,
        "cell_accuracy": 100.0,
        "llm_usage": {
            "requests": {
                "total": 10,
                "successful": 10,
                "failed": 0,
                "retries": 0,
            },
            "tokens": {
                "input": 500,
                "output": 500,
                "thinking": 0,
                "total": 1000,
                "unreported": 0,
            },
            "latency": {
                "total_ms": 500.0,
                "average_request_ms": 50.0,
                "average_case_ms": 100.0,
                "measured_cases": 2,
            },
        },
    }


def make_benchmark_report(
    benchmark_id: str,
    pass_rate: float = 100.0,
    dataset_sha256: str = "a" * 64,
) -> dict[str, Any]:
    """Monta um relatório consolidado válido para comparação."""

    providers = ("gemini", "openai")

    models = {
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4.1-mini",
    }

    provider_runs = {}
    by_provider = {}

    for provider in providers:
        metrics = make_provider_metrics(pass_rate)

        provider_runs[provider] = {
            "execution_status": "completed",
            "report": {
                "run_id": f"{benchmark_id}-{provider}",
                "provider": provider,
                "model": models[provider],
                "summary": {
                    "total_cases": 2,
                    "evaluated_cases": 2,
                    "passed_cases": 2,
                    "failed_cases": 0,
                    "error_cases": 0,
                    "skipped_cases": 0,
                    "pass_rate": pass_rate,
                    "duration_ms": 200.0,
                },
                "results": [],
            },
            "error": None,
            "error_type": None,
        }

        by_provider[provider] = {
            "execution_status": "completed",
            "model": models[provider],
            "run_id": f"{benchmark_id}-{provider}",
            "metrics": metrics,
        }

    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": "2026-11-12T10:00:00+00:00",
        "dataset": {
            "version": 1,
            "source": None,
            "case_count": 2,
            "case_ids": [
                "case-1",
                "case-2",
            ],
            "sha256": dataset_sha256,
        },
        "configuration": {
            "providers": list(providers),
            "provider_count": 2,
            "scenario_count": 2,
            "execution_strategy": "sequential",
            "same_dataset_for_all_providers": True,
        },
        "summary": {
            "provider_count": 2,
            "completed_providers": 2,
            "failed_providers": 0,
            "all_providers_completed": True,
        },
        "provider_runs": provider_runs,
        "comparison": {
            "by_provider": by_provider,
            "by_category": {},
            "by_case": {},
        },
    }


# ==========================================================
# ARGUMENTOS
# ==========================================================


def test_parser_accepts_compare_benchmarks():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--compare-benchmarks",
            "baseline-001",
            "candidate-001",
        ]
    )

    assert args.compare_benchmarks == [
        "baseline-001",
        "candidate-001",
    ]


def test_parser_defaults_compare_benchmarks_to_none():
    parser = cli.build_parser()

    args = parser.parse_args([])

    assert args.compare_benchmarks is None


def test_parser_rejects_compare_benchmarks_with_benchmark_execution():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--compare-benchmarks",
                "baseline-001",
                "candidate-001",
                "--benchmark",
                "gemini",
                "openai",
            ]
        )

    assert exc.value.code == 2


def test_parser_rejects_compare_benchmarks_with_history_listing():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--compare-benchmarks",
                "baseline-001",
                "candidate-001",
                "--list-benchmarks",
            ]
        )

    assert exc.value.code == 2


def test_parser_rejects_compare_benchmarks_with_show_benchmark():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--compare-benchmarks",
                "baseline-001",
                "candidate-001",
                "--show-benchmark",
                "another-id",
            ]
        )

    assert exc.value.code == 2


# ==========================================================
# CÓDIGO DE SAÍDA
# ==========================================================


@pytest.mark.parametrize(
    ("status", "expected_exit_code"),
    [
        ("stable", cli.EXIT_SUCCESS),
        ("improvement", cli.EXIT_SUCCESS),
        ("regression", cli.EXIT_FAILURE),
        ("mixed", cli.EXIT_FAILURE),
        ("incompatible", cli.EXIT_FAILURE),
        ("not_comparable", cli.EXIT_FAILURE),
    ],
)
def test_benchmark_comparison_exit_code(
    status,
    expected_exit_code,
):
    payload = {
        "summary": {
            "status": status,
        },
    }

    assert cli.benchmark_comparison_exit_code(payload) == expected_exit_code


def test_benchmark_comparison_exit_code_rejects_invalid_payload():
    assert cli.benchmark_comparison_exit_code({}) == cli.EXIT_FAILURE

    assert cli.benchmark_comparison_exit_code(None) == cli.EXIT_FAILURE


# ==========================================================
# EXECUÇÃO DA COMPARAÇÃO
# ==========================================================


def test_execute_benchmark_comparison_retrieves_both_reports():
    """Os relatórios devem ser recuperados do histórico pelos IDs."""

    baseline = make_benchmark_report("baseline-001")

    candidate = make_benchmark_report("candidate-001")

    captured = {}

    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            captured.setdefault("requested_ids", []).append(benchmark_id)

            reports = {
                "baseline-001": baseline,
                "candidate-001": candidate,
            }

            return reports.get(benchmark_id)

    result = cli.execute_benchmark_comparison(
        history_store=FakeHistoryStore(),
        baseline_id="baseline-001",
        candidate_id="candidate-001",
    )

    assert captured["requested_ids"] == [
        "baseline-001",
        "candidate-001",
    ]

    assert result["baseline"]["benchmark_id"] == "baseline-001"

    assert result["candidate"]["benchmark_id"] == "candidate-001"

    assert result["summary"]["status"] == "stable"


def test_execute_benchmark_comparison_raises_when_baseline_is_missing():
    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return (
                None
                if benchmark_id == "baseline-001"
                else (make_benchmark_report("candidate-001"))
            )

    with pytest.raises(ValueError, match="baseline-001"):
        cli.execute_benchmark_comparison(
            history_store=FakeHistoryStore(),
            baseline_id="baseline-001",
            candidate_id="candidate-001",
        )


def test_execute_benchmark_comparison_raises_when_candidate_is_missing():
    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return (
                make_benchmark_report("baseline-001")
                if benchmark_id == "baseline-001"
                else None
            )

    with pytest.raises(ValueError, match="candidate-001"):
        cli.execute_benchmark_comparison(
            history_store=FakeHistoryStore(),
            baseline_id="baseline-001",
            candidate_id="candidate-001",
        )


# ==========================================================
# CLASSIFICAÇÃO E APRESENTAÇÃO
# ==========================================================


def test_print_benchmark_comparison_report_displays_summary(
    capsys,
):
    baseline = make_benchmark_report("baseline-001")

    candidate = make_benchmark_report(
        "candidate-001",
        pass_rate=80.0,
    )

    from ai_analyst.evaluation.comparison import BenchmarkComparator

    result = BenchmarkComparator().compare(
        baseline,
        candidate,
    )

    cli.print_benchmark_comparison_report(result)

    captured = capsys.readouterr()

    assert "BENCHMARK COMPARISON" in captured.out

    assert "baseline-001" in captured.out

    assert "candidate-001" in captured.out

    assert "REGRESSION" in captured.out

    assert "GEMINI" in captured.out

    assert "OPENAI" in captured.out


# ==========================================================
# PONTO DE ENTRADA
# ==========================================================


def test_main_compares_benchmarks_without_llm_authorization(
    monkeypatch,
    capsys,
):
    """A comparação histórica não pode exigir autorização de LLM."""

    baseline = make_benchmark_report("baseline-001")

    candidate = make_benchmark_report(
        "candidate-001",
        pass_rate=80.0,
    )

    captured = {}

    class FakeHistoryStore:
        def __init__(self):
            captured["initialized"] = True

        def get_benchmark(self, benchmark_id):
            captured.setdefault("requested_ids", []).append(benchmark_id)

            reports = {
                "baseline-001": baseline,
                "candidate-001": candidate,
            }

            return reports.get(benchmark_id)

    def unexpected_dataset_loading(*args, **kwargs):
        pytest.fail("A comparação histórica não deve carregar o dataset.")

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "load_cases",
        unexpected_dataset_loading,
    )

    exit_code = cli.main(
        [
            "--compare-benchmarks",
            "baseline-001",
            "candidate-001",
        ]
    )

    captured_output = capsys.readouterr()

    assert exit_code == cli.EXIT_FAILURE

    assert captured["initialized"] is True

    assert captured["requested_ids"] == [
        "baseline-001",
        "candidate-001",
    ]

    assert "BENCHMARK COMPARISON" in captured_output.out

    assert "REGRESSION" in captured_output.out


def test_main_returns_failure_when_baseline_does_not_exist(
    monkeypatch,
    capsys,
):
    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return (
                make_benchmark_report("candidate-001")
                if benchmark_id == "candidate-001"
                else None
            )

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--compare-benchmarks",
            "missing-baseline",
            "candidate-001",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_FAILURE

    assert "missing-baseline" in captured.err


def test_main_returns_failure_for_incompatible_datasets(
    monkeypatch,
    capsys,
):
    baseline = make_benchmark_report("baseline-001")

    candidate = make_benchmark_report(
        "candidate-001",
        dataset_sha256="b" * 64,
    )

    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return {
                "baseline-001": baseline,
                "candidate-001": candidate,
            }.get(benchmark_id)

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--compare-benchmarks",
            "baseline-001",
            "candidate-001",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_FAILURE

    assert "INCOMPATIBLE" in captured.out


def test_main_exports_comparison_json(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    baseline = make_benchmark_report("baseline-001")

    candidate = make_benchmark_report("candidate-001")

    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return {
                "baseline-001": baseline,
                "candidate-001": candidate,
            }.get(benchmark_id)

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    output_path = tmp_path / "reports" / "comparison.json"

    exit_code = cli.main(
        [
            "--compare-benchmarks",
            "baseline-001",
            "candidate-001",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS

    assert output_path.exists()

    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_payload["baseline"]["benchmark_id"] == "baseline-001"

    assert saved_payload["candidate"]["benchmark_id"] == "candidate-001"

    assert saved_payload["summary"]["status"] == "stable"


def test_main_comparison_rejects_allow_llm(
    monkeypatch,
):
    class FakeHistoryStore:
        def get_benchmark(self, benchmark_id):
            return make_benchmark_report(benchmark_id)

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--compare-benchmarks",
                "baseline-001",
                "candidate-001",
                "--allow-llm",
            ]
        )

    assert exc.value.code == 2
