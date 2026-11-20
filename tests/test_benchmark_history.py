from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from ai_analyst.evaluation.history import BenchmarkHistory

# ==========================================================
# FIXTURES E FÁBRICAS DE DADOS
# ==========================================================


@pytest.fixture
def history(tmp_path):
    database_path = tmp_path / "test_benchmark_history.sqlite3"

    return BenchmarkHistory(database_path)


def make_benchmark_report(
    benchmark_id: str = "benchmark-001",
    *,
    created_at: str = "2026-01-01T10:00:00+00:00",
    providers: tuple[str, ...] = ("gemini", "openai"),
    models: dict[str, str] | None = None,
    failed_providers: set[str] | None = None,
    dataset_sha256: str = "a" * 64,
    case_count: int = 2,
) -> dict[str, Any]:
    """Gera um relatório válido e completo de benchmark."""

    models = models or {}
    failed_providers = failed_providers or set()

    case_ids = [f"case-{index:03d}" for index in range(1, case_count + 1)]

    provider_runs = {}

    completed_count = 0
    failed_count = 0

    for provider in providers:
        if provider in failed_providers:
            provider_runs[provider] = {
                "execution_status": "error",
                "report": None,
                "error": "Falha simulada no provedor.",
                "error_type": "RuntimeError",
            }

            failed_count += 1

        else:
            model = models.get(
                provider,
                f"{provider}-model",
            )

            provider_runs[provider] = {
                "execution_status": "completed",
                "report": {
                    "run_id": f"{benchmark_id}-{provider}",
                    "provider": provider,
                    "model": model,
                    "summary": {
                        "total_cases": case_count,
                        "passed_cases": case_count,
                        "failed_cases": 0,
                    },
                    "results": [
                        {
                            "case_id": case_id,
                            "status": "passed",
                            "score": 1.0,
                        }
                        for case_id in case_ids
                    ],
                },
                "error": None,
                "error_type": None,
            }

            completed_count += 1

    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "dataset": {
            "version": 1,
            "source": "data/evaluation/evaluation_dataset.json",
            "case_count": case_count,
            "case_ids": case_ids,
            "sha256": dataset_sha256,
        },
        "configuration": {
            "providers": list(providers),
            "provider_count": len(providers),
            "scenario_count": case_count,
            "execution_strategy": "sequential",
            "same_dataset_for_all_providers": True,
        },
        "summary": {
            "provider_count": len(providers),
            "completed_providers": completed_count,
            "failed_providers": failed_count,
            "all_providers_completed": failed_count == 0,
        },
        "provider_runs": provider_runs,
        "comparison": {
            "by_provider": {
                provider: {
                    "execution_status": (
                        "error" if provider in failed_providers else "completed"
                    ),
                }
                for provider in providers
            },
            "by_category": {},
            "by_case": {
                case_id: {
                    "providers": {},
                }
                for case_id in case_ids
            },
        },
    }


# ==========================================================
# INICIALIZAÇÃO
# ==========================================================


def test_initializes_database(history):
    assert history.count_benchmarks() == 0


def test_creates_database_parent_directory(tmp_path):
    database_path = tmp_path / "nested" / "storage" / "benchmark_history.sqlite3"

    history = BenchmarkHistory(database_path)

    assert database_path.exists()
    assert history.count_benchmarks() == 0


def test_initializes_database_tables(history):
    with history._connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }

    assert "benchmark_history" in tables
    assert "benchmark_provider_runs" in tables


def test_initializes_database_indexes(history):
    with history._connect() as connection:
        indexes = {
            row["name"]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index'
                """
            ).fetchall()
        }

    assert "idx_benchmark_history_created_at" in indexes
    assert "idx_benchmark_history_dataset_sha256" in indexes
    assert "idx_benchmark_provider" in indexes
    assert "idx_benchmark_model" in indexes
    assert "idx_benchmark_provider_model" in indexes


# ==========================================================
# PERSISTÊNCIA E RECUPERAÇÃO
# ==========================================================


def test_saves_and_retrieves_complete_benchmark(history):
    report = make_benchmark_report()

    history.save_benchmark(report)

    result = history.get_benchmark("benchmark-001")

    assert result is not None
    assert result == report

    assert result["benchmark_id"] == "benchmark-001"
    assert result["schema_version"] == 1

    assert result["dataset"]["sha256"] == "a" * 64

    assert result["comparison"]["by_case"]["case-001"] == {
        "providers": {},
    }

    assert result["provider_runs"]["gemini"]["report"]["model"] == "gemini-model"
    assert result["provider_runs"]["openai"]["report"]["model"] == "openai-model"


def test_persists_data_between_instances(tmp_path):
    database_path = tmp_path / "persistent_benchmark_history.sqlite3"

    first = BenchmarkHistory(database_path)

    report = make_benchmark_report(
        benchmark_id="persistent-001",
    )

    first.save_benchmark(report)

    second = BenchmarkHistory(database_path)

    assert second.count_benchmarks() == 1
    assert second.get_benchmark("persistent-001") == report


def test_returns_none_for_unknown_benchmark(history):
    assert history.get_benchmark("benchmark-inexistente") is None


def test_rejects_empty_benchmark_id_on_retrieval(history):
    with pytest.raises(ValueError):
        history.get_benchmark(" ")


def test_preserves_failed_provider_execution(history):
    report = make_benchmark_report(
        benchmark_id="partial-benchmark",
        failed_providers={"openai"},
    )

    history.save_benchmark(report)

    result = history.get_benchmark("partial-benchmark")

    assert result is not None

    assert result["summary"]["completed_providers"] == 1
    assert result["summary"]["failed_providers"] == 1
    assert result["summary"]["all_providers_completed"] is False

    assert result["provider_runs"]["gemini"]["execution_status"] == "completed"
    assert result["provider_runs"]["openai"]["execution_status"] == "error"

    assert result["provider_runs"]["openai"]["report"] is None


# ==========================================================
# IMUTABILIDADE E IDENTIFICADORES
# ==========================================================


def test_rejects_duplicate_benchmark_id(history):
    original = make_benchmark_report()

    history.save_benchmark(original)

    modified = make_benchmark_report(
        benchmark_id="benchmark-001",
        dataset_sha256="b" * 64,
    )

    with pytest.raises(
        ValueError,
        match="já está armazenado",
    ):
        history.save_benchmark(modified)

    assert history.get_benchmark("benchmark-001") == original


def test_different_benchmark_ids_can_use_same_dataset(history):
    first = make_benchmark_report(
        benchmark_id="benchmark-001",
        dataset_sha256="a" * 64,
    )

    second = make_benchmark_report(
        benchmark_id="benchmark-002",
        dataset_sha256="a" * 64,
    )

    history.save_benchmark(first)
    history.save_benchmark(second)

    assert history.count_benchmarks() == 2

    assert history.get_benchmark("benchmark-001") == first
    assert history.get_benchmark("benchmark-002") == second


def test_rejects_non_mapping_report(history):
    with pytest.raises(TypeError):
        history.save_benchmark([])


# ==========================================================
# LISTAGEM E PAGINAÇÃO
# ==========================================================


def test_lists_benchmarks_with_pagination(history):
    reports = [
        make_benchmark_report(
            benchmark_id="benchmark-001",
            created_at="2026-01-01T10:00:00+00:00",
        ),
        make_benchmark_report(
            benchmark_id="benchmark-002",
            created_at="2026-01-02T10:00:00+00:00",
        ),
        make_benchmark_report(
            benchmark_id="benchmark-003",
            created_at="2026-01-03T10:00:00+00:00",
        ),
    ]

    for report in reports:
        history.save_benchmark(report)

    first_page = history.list_benchmarks(
        limit=2,
        offset=0,
    )

    assert first_page["pagination"]["total"] == 3
    assert first_page["pagination"]["limit"] == 2
    assert first_page["pagination"]["offset"] == 0
    assert first_page["pagination"]["has_more"] is True

    assert len(first_page["items"]) == 2

    assert first_page["items"][0]["benchmark_id"] == "benchmark-003"
    assert first_page["items"][1]["benchmark_id"] == "benchmark-002"

    second_page = history.list_benchmarks(
        limit=2,
        offset=2,
    )

    assert second_page["pagination"]["total"] == 3
    assert second_page["pagination"]["has_more"] is False

    assert len(second_page["items"]) == 1
    assert second_page["items"][0]["benchmark_id"] == "benchmark-001"


def test_list_returns_summary_without_complete_reports(history):
    history.save_benchmark(
        make_benchmark_report(),
    )

    result = history.list_benchmarks()

    assert len(result["items"]) == 1

    item = result["items"][0]

    assert item["benchmark_id"] == "benchmark-001"
    assert item["schema_version"] == 1

    assert item["dataset"]["case_count"] == 2
    assert item["dataset"]["sha256"] == "a" * 64

    assert item["providers"] == ["gemini", "openai"]

    assert item["models"] == {
        "gemini": "gemini-model",
        "openai": "openai-model",
    }

    assert item["summary"]["provider_count"] == 2
    assert item["summary"]["completed_providers"] == 2
    assert item["summary"]["failed_providers"] == 0
    assert item["summary"]["all_providers_completed"] is True

    assert "provider_runs" not in item
    assert "comparison" not in item
    assert "report_json" not in item


def test_list_represents_failed_provider_without_model(history):
    report = make_benchmark_report(
        failed_providers={"openai"},
    )

    history.save_benchmark(report)

    result = history.list_benchmarks()

    item = result["items"][0]

    assert item["models"]["gemini"] == "gemini-model"
    assert item["models"]["openai"] is None

    assert item["summary"]["completed_providers"] == 1
    assert item["summary"]["failed_providers"] == 1
    assert item["summary"]["all_providers_completed"] is False


# ==========================================================
# FILTROS
# ==========================================================


@pytest.fixture
def populated_history(history):
    """Preenche o banco com diferentes provedores, modelos e datasets."""

    reports = [
        make_benchmark_report(
            benchmark_id="benchmark-001",
            created_at="2026-01-01T10:00:00+00:00",
            providers=("gemini", "openai"),
            models={
                "gemini": "gemini-2.5-flash",
                "openai": "gpt-4.1-mini",
            },
            dataset_sha256="a" * 64,
        ),
        make_benchmark_report(
            benchmark_id="benchmark-002",
            created_at="2026-01-02T10:00:00+00:00",
            providers=("gemini", "anthropic"),
            models={
                "gemini": "gemini-2.5-flash",
                "anthropic": "claude-haiku",
            },
            dataset_sha256="b" * 64,
        ),
        make_benchmark_report(
            benchmark_id="benchmark-003",
            created_at="2026-01-03T10:00:00+00:00",
            providers=("openai", "anthropic"),
            models={
                "openai": "gpt-4.1-mini",
                "anthropic": "claude-haiku",
            },
            dataset_sha256="a" * 64,
        ),
    ]

    for report in reports:
        history.save_benchmark(report)

    return history


def test_filters_benchmarks_by_provider(populated_history):
    result = populated_history.list_benchmarks(
        provider=" GEMINI ",
    )

    assert result["pagination"]["total"] == 2

    assert {item["benchmark_id"] for item in result["items"]} == {
        "benchmark-001",
        "benchmark-002",
    }


def test_filters_benchmarks_by_model(populated_history):
    result = populated_history.list_benchmarks(
        model="CLAUDE-HAIKU",
    )

    assert result["pagination"]["total"] == 2

    assert {item["benchmark_id"] for item in result["items"]} == {
        "benchmark-002",
        "benchmark-003",
    }


def test_filters_benchmarks_by_dataset_fingerprint(populated_history):
    result = populated_history.list_benchmarks(
        dataset_sha256="a" * 64,
    )

    assert result["pagination"]["total"] == 2

    assert {item["benchmark_id"] for item in result["items"]} == {
        "benchmark-001",
        "benchmark-003",
    }


def test_combines_provider_and_model_filters_on_same_run(populated_history):
    # Gemini e Claude aparecem no banco, mas não juntos
    # como provedor/modelo na mesma execução.
    result = populated_history.list_benchmarks(
        provider="gemini",
        model="claude-haiku",
    )

    assert result["pagination"]["total"] == 0
    assert result["items"] == []


def test_combines_provider_model_and_dataset_filters(populated_history):
    result = populated_history.list_benchmarks(
        provider="gemini",
        model="gemini-2.5-flash",
        dataset_sha256="a" * 64,
    )

    assert result["pagination"]["total"] == 1

    assert result["items"][0]["benchmark_id"] == "benchmark-001"


def test_counts_benchmarks_with_filters(populated_history):
    assert populated_history.count_benchmarks() == 3

    assert (
        populated_history.count_benchmarks(
            provider="gemini",
        )
        == 2
    )

    assert (
        populated_history.count_benchmarks(
            model="claude-haiku",
        )
        == 2
    )

    assert (
        populated_history.count_benchmarks(
            dataset_sha256="a" * 64,
        )
        == 2
    )

    assert (
        populated_history.count_benchmarks(
            provider="gemini",
            model="claude-haiku",
        )
        == 0
    )


def test_count_returns_zero_for_unknown_provider(history):
    assert history.count_benchmarks(provider="unknown") == 0


# ==========================================================
# VALIDAÇÃO DE PAGINAÇÃO E FILTROS
# ==========================================================


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        101,
        True,
    ],
)
def test_rejects_invalid_limit(history, limit):
    with pytest.raises(ValueError):
        history.list_benchmarks(limit=limit)


@pytest.mark.parametrize(
    "offset",
    [
        -1,
        -10,
        True,
    ],
)
def test_rejects_invalid_offset(history, offset):
    with pytest.raises(ValueError):
        history.list_benchmarks(offset=offset)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", ""),
        ("provider", " "),
        ("model", ""),
        ("model", " "),
        ("dataset_sha256", ""),
        ("dataset_sha256", "invalid-hash"),
    ],
)
def test_rejects_invalid_filters(history, field, value):
    with pytest.raises(ValueError):
        history.list_benchmarks(
            **{
                field: value,
            }
        )


# ==========================================================
# VALIDAÇÃO DO CONTRATO DO RELATÓRIO
# ==========================================================


@pytest.mark.parametrize(
    "invalid_field",
    [
        "benchmark_id",
        "created_at",
        "created_at_without_timezone",
        "schema_version",
        "dataset_version",
        "case_count",
        "case_ids",
        "dataset_sha256",
        "providers",
        "duplicate_providers",
        "provider_count",
        "summary_counts",
        "summary_completion",
        "provider_runs",
        "execution_status",
    ],
)
def test_rejects_invalid_benchmark_report(history, invalid_field):
    report = make_benchmark_report()

    if invalid_field == "benchmark_id":
        report["benchmark_id"] = " "

    elif invalid_field == "created_at":
        report["created_at"] = "data-invalida"

    elif invalid_field == "created_at_without_timezone":
        report["created_at"] = "2026-01-01T10:00:00"

    elif invalid_field == "schema_version":
        report["schema_version"] = 0

    elif invalid_field == "dataset_version":
        report["dataset"]["version"] = 0

    elif invalid_field == "case_count":
        report["dataset"]["case_count"] = 3

    elif invalid_field == "case_ids":
        report["dataset"]["case_ids"] = [
            "case-001",
            "case-001",
        ]

    elif invalid_field == "dataset_sha256":
        report["dataset"]["sha256"] = "not-a-sha256"

    elif invalid_field == "providers":
        report["configuration"]["providers"] = ["gemini"]

    elif invalid_field == "duplicate_providers":
        report["configuration"]["providers"] = [
            "gemini",
            "gemini",
        ]

    elif invalid_field == "provider_count":
        report["configuration"]["provider_count"] = 3

    elif invalid_field == "summary_counts":
        report["summary"]["completed_providers"] = 1

    elif invalid_field == "summary_completion":
        report["summary"]["all_providers_completed"] = False

    elif invalid_field == "provider_runs":
        report["provider_runs"].pop("openai")

    elif invalid_field == "execution_status":
        report["provider_runs"]["gemini"]["execution_status"] = "pending"

    with pytest.raises(ValueError):
        history.save_benchmark(report)

    assert history.count_benchmarks() == 0


def test_rejects_non_json_serializable_report(history):
    report = make_benchmark_report()

    report["unexpected_value"] = object()

    with pytest.raises(ValueError):
        history.save_benchmark(report)

    assert history.count_benchmarks() == 0


def test_rejects_nan_in_report(history):
    report = make_benchmark_report()

    report["comparison"]["unexpected_metric"] = float("nan")

    with pytest.raises(ValueError):
        history.save_benchmark(report)

    assert history.count_benchmarks() == 0


# ==========================================================
# INTEGRIDADE TRANSACIONAL
# ==========================================================


def test_rolls_back_benchmark_when_provider_insert_fails(history):
    """A falha de uma inserção deve reverter toda a transação."""

    with history._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_openai_insert
            BEFORE INSERT ON benchmark_provider_runs
            WHEN NEW.provider = 'openai'
            BEGIN
                SELECT RAISE(ABORT, 'Falha simulada de integridade.');
            END;
            """
        )

    report = make_benchmark_report(
        benchmark_id="transaction-001",
    )

    with pytest.raises(sqlite3.IntegrityError):
        history.save_benchmark(report)

    assert history.count_benchmarks() == 0

    with history._connect() as connection:
        provider_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM benchmark_provider_runs
            """
        ).fetchone()[0]

    assert provider_count == 0


def test_persists_all_provider_records_in_same_transaction(history):
    report = make_benchmark_report(
        providers=("gemini", "openai", "anthropic"),
    )

    history.save_benchmark(report)

    with history._connect() as connection:
        records = connection.execute(
            """
            SELECT provider, execution_status
            FROM benchmark_provider_runs
            WHERE benchmark_id = ?
            ORDER BY provider
            """,
            ("benchmark-001",),
        ).fetchall()

    assert len(records) == 3

    assert [(record["provider"], record["execution_status"]) for record in records] == [
        ("anthropic", "completed"),
        ("gemini", "completed"),
        ("openai", "completed"),
    ]


# ==========================================================
# CORRUPÇÃO E RECUPERAÇÃO
# ==========================================================


def test_reports_corrupted_json_when_retrieving_benchmark(history):
    history.save_benchmark(
        make_benchmark_report(),
    )

    with history._connect() as connection:
        connection.execute(
            """
            UPDATE benchmark_history
            SET report_json = ?
            WHERE benchmark_id = ?
            """,
            (
                "{json-invalido",
                "benchmark-001",
            ),
        )

    with pytest.raises(
        ValueError,
        match="corrompido",
    ):
        history.get_benchmark("benchmark-001")


def test_reports_corrupted_provider_list_in_summary(history):
    history.save_benchmark(
        make_benchmark_report(),
    )

    with history._connect() as connection:
        connection.execute(
            """
            UPDATE benchmark_history
            SET providers_json = ?
            WHERE benchmark_id = ?
            """,
            (
                "{json-invalido",
                "benchmark-001",
            ),
        )

    with pytest.raises(
        ValueError,
        match="corrompida",
    ):
        history.list_benchmarks()


# ==========================================================
# EXCLUSÃO
# ==========================================================


def test_deletes_benchmark(history):
    history.save_benchmark(
        make_benchmark_report(),
    )

    assert history.delete_benchmark("benchmark-001") is True

    assert history.get_benchmark("benchmark-001") is None

    assert history.count_benchmarks() == 0


def test_deleting_benchmark_removes_provider_metadata(history):
    history.save_benchmark(
        make_benchmark_report(
            providers=("gemini", "openai", "anthropic"),
        )
    )

    assert history.delete_benchmark("benchmark-001") is True

    with history._connect() as connection:
        result = connection.execute(
            """
            SELECT COUNT(*)
            FROM benchmark_provider_runs
            WHERE benchmark_id = ?
            """,
            ("benchmark-001",),
        ).fetchone()[0]

    assert result == 0


def test_returns_false_when_deleting_unknown_benchmark(history):
    assert history.delete_benchmark("inexistente") is False


def test_rejects_empty_id_when_deleting_benchmark(history):
    with pytest.raises(ValueError):
        history.delete_benchmark(" ")


# ==========================================================
# CICLO DE VIDA DAS CONEXÕES
# ==========================================================


def test_closes_connection_after_operation(history):
    with history._connect() as connection:
        pass

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_uses_foreign_key_enforcement(history):
    with history._connect() as connection:
        foreign_keys_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert foreign_keys_enabled == 1


def test_independent_database_paths_are_isolated(tmp_path):
    first = BenchmarkHistory(
        tmp_path / "first.sqlite3",
    )

    second = BenchmarkHistory(
        tmp_path / "second.sqlite3",
    )

    first.save_benchmark(
        make_benchmark_report(),
    )

    assert first.count_benchmarks() == 1
    assert second.count_benchmarks() == 0

    assert second.get_benchmark("benchmark-001") is None
