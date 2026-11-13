"""Testes de integração da CLI com o contrato real do BenchmarkHistory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_analyst.evaluation import cli

# ==========================================================
# FÁBRICAS
# ==========================================================


def make_history_entry(
    benchmark_id: str = "benchmark-test-001",
) -> dict[str, Any]:
    """Monta um registro no formato real retornado pelo SQLite."""

    return {
        "benchmark_id": benchmark_id,
        "created_at": "2026-11-12T10:00:00+00:00",
        "schema_version": 1,
        "dataset": {
            "version": 1,
            "source": None,
            "case_count": 2,
            "sha256": "a" * 64,
        },
        "providers": [
            "gemini",
            "openai",
        ],
        "models": {
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4.1-mini",
        },
        "summary": {
            "provider_count": 2,
            "completed_providers": 2,
            "failed_providers": 0,
            "all_providers_completed": True,
        },
    }


def make_history_result(
    entries: list[dict[str, Any]] | None = None,
    *,
    total: int = 1,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Monta o retorno real de BenchmarkHistory.list_benchmarks()."""

    if entries is None:
        entries = [make_history_entry()]

    return {
        "items": entries,
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(entries) < total,
        },
    }


# ==========================================================
# CONSULTA DO HISTÓRICO
# ==========================================================


def test_list_benchmark_history_accepts_real_history_contract():
    """A CLI deve consumir items e pagination sem segunda consulta."""

    expected_result = make_history_result()

    class FakeHistoryStore:
        def list_benchmarks(
            self,
            limit,
            offset,
            provider=None,
            model=None,
        ):
            return expected_result

    result = cli.list_benchmark_history(
        history_store=FakeHistoryStore(),
        limit=20,
        offset=0,
    )

    assert result["items"] == expected_result["items"]

    assert result["pagination"] == expected_result["pagination"]

    assert result["filters"] == {
        "provider": None,
        "model": None,
    }


def test_list_benchmark_history_preserves_filters_and_pagination():
    """Filtros e paginação devem preservar o contrato do armazenamento."""

    expected_result = make_history_result(
        entries=[
            make_history_entry("benchmark-filtered-001"),
        ],
        total=11,
        limit=5,
        offset=10,
    )

    captured = {}

    class FakeHistoryStore:
        def list_benchmarks(
            self,
            limit,
            offset,
            provider=None,
            model=None,
        ):
            captured["arguments"] = {
                "limit": limit,
                "offset": offset,
                "provider": provider,
                "model": model,
            }

            return expected_result

    result = cli.list_benchmark_history(
        history_store=FakeHistoryStore(),
        limit=5,
        offset=10,
        provider="gemini",
        model="gemini-2.5-flash",
    )

    assert captured["arguments"] == {
        "limit": 5,
        "offset": 10,
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    }

    assert result["items"] == expected_result["items"]

    assert result["pagination"]["total"] == 11

    assert result["pagination"]["has_more"] is False

    assert result["filters"] == {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
    }


# ==========================================================
# FORMATAÇÃO
# ==========================================================


def test_print_benchmark_history_displays_nested_fields(capsys):
    """A formatação deve ler os campos aninhados do histórico."""

    entry = make_history_entry()

    cli.print_benchmark_history(
        entries=[entry],
        total=1,
        limit=20,
        offset=0,
    )

    captured = capsys.readouterr()

    assert "benchmark-test-001" in captured.out

    assert "2026-11-12T10:00:00+00:00" in captured.out

    assert "Cenários: 2" in captured.out

    assert "Dataset SHA-256" in captured.out

    assert "a" * 64 in captured.out

    assert "gemini" in captured.out

    assert "openai" in captured.out

    assert "gemini-2.5-flash" in captured.out

    assert "gpt-4.1-mini" in captured.out

    assert "2 concluídas / 0 com falha" in captured.out


# ==========================================================
# PONTO DE ENTRADA
# ==========================================================


def test_main_lists_benchmarks_using_real_history_contract(
    monkeypatch,
    capsys,
):
    """A CLI deve listar registros no formato real do SQLite."""

    expected_result = make_history_result()

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
            captured["arguments"] = {
                "limit": limit,
                "offset": offset,
                "provider": provider,
                "model": model,
            }

            return expected_result

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

    assert captured["initialized"] is True

    assert captured["arguments"] == {
        "limit": 20,
        "offset": 0,
        "provider": None,
        "model": None,
    }

    assert "benchmark-test-001" in captured_output.out

    assert "gemini-2.5-flash" in captured_output.out


def test_main_exports_history_with_pagination(
    monkeypatch,
    tmp_path: Path,
):
    """A exportação deve preservar a estrutura original do histórico."""

    expected_result = make_history_result(
        total=5,
        limit=2,
        offset=0,
    )

    class FakeHistoryStore:
        def list_benchmarks(
            self,
            limit,
            offset,
            provider=None,
            model=None,
        ):
            return expected_result

    monkeypatch.setattr(
        cli,
        "BenchmarkHistory",
        FakeHistoryStore,
        raising=False,
    )

    output_path = tmp_path / "reports" / "history.json"

    exit_code = cli.main(
        [
            "--list-benchmarks",
            "--limit",
            "2",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == cli.EXIT_SUCCESS

    assert output_path.exists()

    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(saved_payload["items"]) == 1

    assert saved_payload["items"][0]["benchmark_id"] == ("benchmark-test-001")

    assert saved_payload["pagination"]["total"] == 5

    assert saved_payload["pagination"]["limit"] == 2

    assert saved_payload["pagination"]["offset"] == 0

    assert saved_payload["pagination"]["has_more"] is True
