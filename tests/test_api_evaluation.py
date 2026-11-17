from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from ai_analyst.api import evaluation as evaluation_api
from ai_analyst.api.main import app
from ai_analyst.evaluation.history import BenchmarkHistory

client = TestClient(app)


def create_test_history(monkeypatch, tmp_path):
    """Substitui o histórico global por um banco SQLite temporário."""

    history = BenchmarkHistory(tmp_path / "test_benchmark_history.sqlite3")

    monkeypatch.setattr(
        evaluation_api,
        "benchmark_history",
        history,
    )

    return history


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
    """Cria um relatório artificial válido para persistência."""

    models = models or {}
    failed_providers = failed_providers or set()

    case_ids = [f"case-{index:03d}" for index in range(1, case_count + 1)]

    provider_runs: dict[str, Any] = {}

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


def test_benchmark_history_lists_records(monkeypatch, tmp_path):
    history = create_test_history(monkeypatch, tmp_path)

    history.save_benchmark(
        make_benchmark_report(
            "benchmark-001",
            created_at="2026-11-17T10:00:00+00:00",
        )
    )

    history.save_benchmark(
        make_benchmark_report(
            "benchmark-002",
            created_at="2026-11-17T11:00:00+00:00",
        )
    )

    response = client.get("/evaluation/benchmarks")

    assert response.status_code == 200

    data = response.json()

    assert data["pagination"]["total"] == 2
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["has_more"] is False

    assert len(data["items"]) == 2

    returned_ids = {item["benchmark_id"] for item in data["items"]}

    assert returned_ids == {
        "benchmark-001",
        "benchmark-002",
    }

    assert "provider_runs" not in data["items"][0]


def test_benchmark_history_supports_pagination(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)

    for index in range(3):
        history.save_benchmark(
            make_benchmark_report(
                f"benchmark-{index}",
                created_at=f"2026-11-17T10:0{index}:00+00:00",
            )
        )

    first_response = client.get(
        "/evaluation/benchmarks",
        params={"limit": 2, "offset": 0},
    )

    second_response = client.get(
        "/evaluation/benchmarks",
        params={"limit": 2, "offset": 2},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    assert len(first_data["items"]) == 2
    assert len(second_data["items"]) == 1

    assert first_data["pagination"]["total"] == 3
    assert first_data["pagination"]["has_more"] is True

    assert second_data["pagination"]["offset"] == 2
    assert second_data["pagination"]["has_more"] is False


def test_benchmark_history_filters_by_provider_model_and_dataset(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)

    history.save_benchmark(
        make_benchmark_report(
            "benchmark-gemini",
            providers=("gemini", "openai"),
            models={
                "gemini": "gemini-2.5-flash",
                "openai": "gpt-4.1-mini",
            },
            dataset_sha256="a" * 64,
        )
    )

    history.save_benchmark(
        make_benchmark_report(
            "benchmark-other",
            providers=("gemini", "anthropic"),
            models={
                "gemini": "gemini-other",
                "anthropic": "claude-haiku",
            },
            dataset_sha256="b" * 64,
        )
    )

    response = client.get(
        "/evaluation/benchmarks",
        params={
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "dataset_sha256": "a" * 64,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["pagination"]["total"] == 1
    assert data["items"][0]["benchmark_id"] == "benchmark-gemini"


def test_benchmark_history_rejects_invalid_dataset_hash(
    monkeypatch,
    tmp_path,
):
    create_test_history(monkeypatch, tmp_path)

    response = client.get(
        "/evaluation/benchmarks",
        params={"dataset_sha256": "invalid"},
    )

    assert response.status_code == 400
    assert "SHA-256" in response.json()["detail"]


def test_benchmark_detail_returns_complete_report(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)

    report = make_benchmark_report("benchmark-detail")

    history.save_benchmark(report)

    response = client.get("/evaluation/benchmarks/benchmark-detail")

    assert response.status_code == 200

    data = response.json()

    assert data["benchmark_id"] == "benchmark-detail"
    assert data["dataset"]["sha256"] == "a" * 64
    assert "provider_runs" in data
    assert "comparison" in data


def test_benchmark_detail_sanitizes_internal_errors(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)

    report = make_benchmark_report(
        "benchmark-failed",
        failed_providers={"gemini"},
    )

    history.save_benchmark(report)

    response = client.get("/evaluation/benchmarks/benchmark-failed")

    assert response.status_code == 200

    data = response.json()

    assert data["provider_runs"]["gemini"]["error"] == evaluation_api.SAFE_ERROR_DETAIL

    assert data["provider_runs"]["gemini"]["error_type"] == "RuntimeError"


def test_benchmark_detail_returns_404_for_missing_id(
    monkeypatch,
    tmp_path,
):
    create_test_history(monkeypatch, tmp_path)

    response = client.get("/evaluation/benchmarks/does-not-exist")

    assert response.status_code == 404


def test_benchmark_comparison_delegates_to_comparator(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)

    baseline = make_benchmark_report("baseline-001")
    candidate = make_benchmark_report("candidate-001")

    history.save_benchmark(baseline)
    history.save_benchmark(candidate)

    class FakeComparator:
        def compare(self, *, baseline, candidate):
            assert baseline["benchmark_id"] == "baseline-001"
            assert candidate["benchmark_id"] == "candidate-001"

            return {
                "schema_version": 1,
                "compatibility": {
                    "comparable": False,
                },
                "summary": {
                    "status": "incompatible",
                },
                "providers": {},
            }

    monkeypatch.setattr(
        evaluation_api,
        "benchmark_comparator",
        FakeComparator(),
    )

    response = client.post(
        "/evaluation/benchmarks/compare",
        json={
            "baseline_id": "baseline-001",
            "candidate_id": "candidate-001",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["compatibility"]["comparable"] is False
    assert data["summary"]["status"] == "incompatible"


def test_benchmark_comparison_returns_404_for_missing_baseline(
    monkeypatch,
    tmp_path,
):
    create_test_history(monkeypatch, tmp_path)

    response = client.post(
        "/evaluation/benchmarks/compare",
        json={
            "baseline_id": "missing-baseline",
            "candidate_id": "missing-candidate",
        },
    )

    assert response.status_code == 404
    assert "referência" in response.json()["detail"]


def test_benchmark_comparison_returns_404_for_missing_candidate(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)

    history.save_benchmark(make_benchmark_report("baseline-001"))

    response = client.post(
        "/evaluation/benchmarks/compare",
        json={
            "baseline_id": "baseline-001",
            "candidate_id": "missing-candidate",
        },
    )

    assert response.status_code == 404
    assert "candidato" in response.json()["detail"]


def test_benchmark_comparison_rejects_same_id(
    monkeypatch,
    tmp_path,
):
    create_test_history(monkeypatch, tmp_path)

    response = client.post(
        "/evaluation/benchmarks/compare",
        json={
            "baseline_id": "same-benchmark",
            "candidate_id": "same-benchmark",
        },
    )

    assert response.status_code == 400


def test_benchmark_comparison_rejects_whitespace_ids():
    response = client.post(
        "/evaluation/benchmarks/compare",
        json={
            "baseline_id": "   ",
            "candidate_id": "candidate-001",
        },
    )

    assert response.status_code == 400
