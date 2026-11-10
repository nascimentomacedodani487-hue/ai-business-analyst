"""Testes da interface de linha de comando do AI Evaluation Engine."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from ai_analyst.evaluation import cli
from ai_analyst.evaluation.dataset import load_evaluation_dataset

# ==========================================================
# FIXTURES E DADOS DE TESTE
# ==========================================================


@pytest.fixture
def cases():
    """Carrega o dataset real sem executar consultas ou LLMs."""

    return load_evaluation_dataset()


@pytest.fixture
def sample_case(cases):
    """Disponibiliza um cenário válido do dataset."""

    return cases[0]


class FakeSQLExecutor:
    """Executor controlado para testes sem depender do DuckDB."""

    def __init__(
        self,
        result=None,
        error=None,
    ):
        self.result = result
        self.error = error
        self.executed_sql = []

    def execute(self, sql):
        self.executed_sql.append(sql)

        if self.error is not None:
            raise self.error

        return self.result


def make_validation_payload(case_id="total_orders"):
    """Monta um relatório fictício de validação SQL."""

    return {
        "mode": "reference_validation",
        "database": "DuckDB",
        "summary": {
            "total_cases": 1,
            "passed_cases": 1,
            "error_cases": 0,
            "pass_rate": 100.0,
        },
        "results": [
            {
                "case_id": case_id,
                "category": "volume",
                "status": "passed",
                "row_count": 1,
                "columns": ["total_orders"],
                "required_columns": ["total_orders"],
                "schema_validation": "validated",
                "duration_ms": 10.0,
                "error": None,
            }
        ],
    }


def make_evaluation_payload(case_id="total_orders"):
    """Monta um relatório fictício de avaliação completa."""

    return {
        "run_id": "test-run-123",
        "provider": "gemini",
        "model": "test-model",
        "started_at": "2026-11-09T10:00:00",
        "finished_at": "2026-11-09T10:00:01",
        "summary": {
            "total_cases": 1,
            "evaluated_cases": 1,
            "passed_cases": 1,
            "failed_cases": 0,
            "error_cases": 0,
            "skipped_cases": 0,
            "pass_rate": 100.0,
            "duration_ms": 1000.0,
        },
        "metadata": {
            "metrics": {
                "summary": {
                    "average_score": 1.0,
                    "average_duration_ms": 1000.0,
                    "hit_rate": 0.0,
                }
            }
        },
        "results": [
            {
                "case_id": case_id,
                "question": "Pergunta de teste",
                "status": "passed",
                "score": 1.0,
                "expected_rows": [{"total_orders": 60000}],
                "actual_rows": [{"total_orders": 60000}],
                "differences": [],
                "analysis_id": "analysis-test",
                "provider": "gemini",
                "model": "test-model",
                "duration_ms": 1000.0,
                "cache_hit": False,
                "error": None,
                "metadata": {},
            }
        ],
    }


# ==========================================================
# ARGUMENTOS
# ==========================================================


def test_build_parser_has_expected_defaults():
    parser = cli.build_parser()

    args = parser.parse_args([])

    assert args.list_cases is False
    assert args.validate_only is False
    assert args.provider == "gemini"
    assert args.dataset is None
    assert args.case_ids is None
    assert args.output is None
    assert args.allow_llm is False
    assert args.log_level == "WARNING"


def test_build_parser_accepts_multiple_cases():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--validate-only",
            "--case",
            "total_orders",
            "--case",
            "payment_distribution",
        ]
    )

    assert args.case_ids == [
        "total_orders",
        "payment_distribution",
    ]


def test_build_parser_accepts_dataset_and_output():
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "--dataset",
            "custom_dataset.json",
            "--output",
            "reports/result.json",
        ]
    )

    assert args.dataset == Path("custom_dataset.json")
    assert args.output == Path("reports/result.json")


def test_build_parser_restricts_providers():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(
            [
                "--provider",
                "unsupported",
            ]
        )

    assert exc.value.code == 2


# ==========================================================
# CARREGAMENTO E SELEÇÃO DO DATASET
# ==========================================================


def test_load_cases_loads_real_dataset():
    cases = cli.load_cases()

    assert len(cases) == 10

    assert all(case.case_id for case in cases)

    assert len({case.case_id for case in cases}) == len(cases)

    case_ids = {case.case_id for case in cases}

    assert "operational_overview" in case_ids


def test_load_cases_rejects_empty_dataset(monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_evaluation_dataset",
        lambda path=None: [],
    )

    with pytest.raises(
        ValueError,
        match="não contém cenários",
    ):
        cli.load_cases()


def test_load_cases_rejects_duplicate_ids(monkeypatch, cases):
    duplicated_cases = [
        cases[0],
        cases[0],
    ]

    monkeypatch.setattr(
        cli,
        "load_evaluation_dataset",
        lambda path=None: duplicated_cases,
    )

    with pytest.raises(
        ValueError,
        match="duplicados",
    ):
        cli.load_cases()


def test_select_cases_preserves_dataset_order(cases):
    selected = cli.select_cases(
        cases,
        [
            "payment_distribution",
            "total_orders",
        ],
    )

    assert [case.case_id for case in selected] == [
        "total_orders",
        "payment_distribution",
    ]


def test_select_cases_removes_repeated_requested_ids(cases):
    selected = cli.select_cases(
        cases,
        [
            "total_orders",
            "total_orders",
        ],
    )

    assert len(selected) == 1
    assert selected[0].case_id == "total_orders"


def test_select_cases_returns_all_cases_when_no_filter(cases):
    selected = cli.select_cases(
        cases,
        None,
    )

    assert selected == cases


def test_select_cases_rejects_unknown_ids(cases):
    with pytest.raises(
        ValueError,
        match="não encontrado",
    ):
        cli.select_cases(
            cases,
            ["unknown_case"],
        )


# ==========================================================
# SERIALIZAÇÃO JSON
# ==========================================================


def test_json_default_converts_decimal():
    result = cli.json_default(Decimal("125.75"))

    assert result == 125.75


def test_json_default_converts_datetime():
    value = datetime(
        2026,
        11,
        9,
        12,
        30,
        tzinfo=timezone.utc,
    )

    result = cli.json_default(value)

    assert result == "2026-11-09T12:30:00+00:00"


def test_json_default_converts_path():
    original = Path("reports/result.json")

    result = cli.json_default(original)

    assert isinstance(result, str)
    assert Path(result) == original


def test_json_default_converts_sets():
    result = cli.json_default({"b", "a", "c"})

    assert result == ["a", "b", "c"]


def test_json_default_rejects_unsupported_types():
    with pytest.raises(
        TypeError,
        match="Tipo não suportado",
    ):
        cli.json_default(object())


def test_write_json_report_creates_valid_file(tmp_path):
    output_path = tmp_path / "reports" / "result.json"

    payload = {
        "provider": "gemini",
        "value": Decimal("150.50"),
        "cases": 10,
    }

    saved_path = cli.write_json_report(
        output_path,
        payload,
    )

    assert saved_path == output_path
    assert output_path.exists()

    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded["provider"] == "gemini"
    assert loaded["value"] == 150.50
    assert loaded["cases"] == 10


# ==========================================================
# VALIDAÇÃO DOS SQLs DE REFERÊNCIA
# ==========================================================


def test_validate_reference_cases_passes(sample_case):
    columns = sample_case.required_columns or ("total_orders",)

    reference_rows = [{column: 100 for column in columns}]

    executor = FakeSQLExecutor(result=reference_rows)

    payload = cli.validate_reference_cases(
        cases=[sample_case],
        executor=executor,
    )

    assert len(executor.executed_sql) == 1

    assert payload["summary"]["total_cases"] == 1
    assert payload["summary"]["passed_cases"] == 1
    assert payload["summary"]["error_cases"] == 0
    assert payload["summary"]["pass_rate"] == 100.0

    result = payload["results"][0]

    assert result["case_id"] == sample_case.case_id
    assert result["status"] == "passed"
    assert result["row_count"] == 1
    assert result["error"] is None


def test_validate_reference_cases_detects_missing_columns(sample_case):
    case = replace(
        sample_case,
        required_columns=("required_column",),
    )

    executor = FakeSQLExecutor(result=[{"unexpected_column": 100}])

    payload = cli.validate_reference_cases(
        cases=[case],
        executor=executor,
    )

    result = payload["results"][0]

    assert result["status"] == "error"
    assert "Colunas obrigatórias ausentes" in result["error"]

    assert payload["summary"]["passed_cases"] == 0
    assert payload["summary"]["error_cases"] == 1
    assert payload["summary"]["pass_rate"] == 0.0


def test_validate_reference_cases_handles_sql_exception(sample_case):
    executor = FakeSQLExecutor(error=RuntimeError("Erro de execução SQL"))

    payload = cli.validate_reference_cases(
        cases=[sample_case],
        executor=executor,
    )

    result = payload["results"][0]

    assert result["status"] == "error"
    assert result["error"] == "Erro de execução SQL"
    assert result["error_type"] == "RuntimeError"

    assert payload["summary"]["error_cases"] == 1


def test_validate_reference_cases_rejects_non_list_result(sample_case):
    executor = FakeSQLExecutor(result={"total_orders": 60000})

    payload = cli.validate_reference_cases(
        cases=[sample_case],
        executor=executor,
    )

    result = payload["results"][0]

    assert result["status"] == "error"
    assert "não retornou uma lista" in result["error"]


def test_validate_reference_cases_rejects_invalid_rows(sample_case):
    executor = FakeSQLExecutor(result=[123])

    payload = cli.validate_reference_cases(
        cases=[sample_case],
        executor=executor,
    )

    result = payload["results"][0]

    assert result["status"] == "error"
    assert "formato inválido" in result["error"]


# ==========================================================
# CÓDIGOS DE SAÍDA
# ==========================================================


def test_validation_exit_code_returns_success():
    payload = make_validation_payload()

    assert cli.validation_exit_code(payload) == cli.EXIT_SUCCESS


def test_validation_exit_code_returns_failure():
    payload = make_validation_payload()

    payload["summary"]["error_cases"] = 1

    assert cli.validation_exit_code(payload) == cli.EXIT_FAILURE


def test_evaluation_exit_code_returns_success():
    payload = make_evaluation_payload()

    assert cli.evaluation_exit_code(payload) == cli.EXIT_SUCCESS


@pytest.mark.parametrize(
    "summary_field",
    [
        "failed_cases",
        "error_cases",
        "skipped_cases",
    ],
)
def test_evaluation_exit_code_returns_failure(summary_field):
    payload = make_evaluation_payload()

    payload["summary"][summary_field] = 1

    assert cli.evaluation_exit_code(payload) == cli.EXIT_FAILURE


# ==========================================================
# PONTO DE ENTRADA DA CLI
# ==========================================================


def test_main_lists_cases_without_running_evaluation(
    monkeypatch,
    cases,
    capsys,
):
    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("A listagem não deve executar avaliações.")

    monkeypatch.setattr(
        cli,
        "execute_evaluation",
        unexpected_evaluation,
    )

    exit_code = cli.main(
        [
            "--list-cases",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert "Total de cenários: 10" in captured.out
    assert "total_orders" in captured.out
    assert "payment_distribution" in captured.out
    assert "operational_overview" in captured.out


def test_main_requires_explicit_llm_authorization():
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--provider",
                "gemini",
            ]
        )

    assert exc.value.code == 2


def test_main_rejects_llm_authorization_in_validation_mode():
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--validate-only",
                "--allow-llm",
            ]
        )

    assert exc.value.code == 2


def test_main_validation_does_not_execute_llm(
    monkeypatch,
    cases,
    capsys,
):
    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    payload = make_validation_payload()

    monkeypatch.setattr(
        cli,
        "validate_reference_cases",
        lambda selected_cases: payload,
    )

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("O modo validate-only não deve executar LLMs.")

    monkeypatch.setattr(
        cli,
        "execute_evaluation",
        unexpected_evaluation,
    )

    exit_code = cli.main(
        [
            "--validate-only",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == cli.EXIT_SUCCESS
    assert "VALIDAÇÃO DE REFERÊNCIAS" in captured.out
    assert "100.00%" in captured.out


def test_main_validation_exports_json(
    monkeypatch,
    cases,
    tmp_path,
):
    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    payload = make_validation_payload()

    monkeypatch.setattr(
        cli,
        "validate_reference_cases",
        lambda selected_cases: payload,
    )

    output_path = tmp_path / "validation.json"

    exit_code = cli.main(
        [
            "--validate-only",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert output_path.exists()

    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_payload["mode"] == "reference_validation"
    assert saved_payload["summary"]["passed_cases"] == 1


def test_main_runs_evaluation_with_explicit_authorization(
    monkeypatch,
    cases,
):
    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    expected_payload = make_evaluation_payload()

    captured = {}

    def fake_execute(
        cases,
        provider,
        dataset_path=None,
    ):
        captured["cases"] = cases
        captured["provider"] = provider
        captured["dataset_path"] = dataset_path

        return expected_payload

    monkeypatch.setattr(
        cli,
        "execute_evaluation",
        fake_execute,
    )

    monkeypatch.setattr(
        cli,
        "print_evaluation_report",
        lambda payload: None,
    )

    exit_code = cli.main(
        [
            "--provider",
            "anthropic",
            "--allow-llm",
            "--case",
            "total_orders",
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert captured["provider"] == "anthropic"
    assert [case.case_id for case in captured["cases"]] == ["total_orders"]


def test_main_exports_complete_evaluation_report(
    monkeypatch,
    cases,
    tmp_path,
):
    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    expected_payload = make_evaluation_payload()

    monkeypatch.setattr(
        cli,
        "execute_evaluation",
        lambda **kwargs: expected_payload,
    )

    monkeypatch.setattr(
        cli,
        "print_evaluation_report",
        lambda payload: None,
    )

    output_path = tmp_path / "reports" / "evaluation.json"

    exit_code = cli.main(
        [
            "--allow-llm",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS
    assert output_path.exists()

    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved_payload["run_id"] == "test-run-123"
    assert saved_payload["provider"] == "gemini"
    assert saved_payload["summary"]["passed_cases"] == 1


def test_main_returns_error_when_dataset_loading_fails(
    monkeypatch,
):
    def fail_loading(path=None):
        raise RuntimeError("Dataset indisponível")

    monkeypatch.setattr(
        cli,
        "load_cases",
        fail_loading,
    )

    exit_code = cli.main(
        [
            "--list-cases",
        ]
    )

    assert exit_code == cli.EXIT_ERROR


def test_main_returns_error_when_evaluation_fails(
    monkeypatch,
    cases,
):
    monkeypatch.setattr(
        cli,
        "load_cases",
        lambda path=None: cases,
    )

    def fail_evaluation(**kwargs):
        raise RuntimeError("Falha simulada na avaliação")

    monkeypatch.setattr(
        cli,
        "execute_evaluation",
        fail_evaluation,
    )

    exit_code = cli.main(
        [
            "--allow-llm",
        ]
    )

    assert exit_code == cli.EXIT_ERROR
