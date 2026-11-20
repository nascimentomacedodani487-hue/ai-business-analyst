from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from ai_analyst.api import main as api_main
from ai_analyst.data.history import AnalysisHistory
from ai_analyst.llm.metrics import LLMMetrics

client = TestClient(api_main.app)


@pytest.fixture
def metrics():
    return LLMMetrics()


@pytest.fixture
def api_metrics(monkeypatch):
    """Fornece métricas isoladas para os testes dos endpoints."""

    instance = LLMMetrics()

    monkeypatch.setattr(
        api_main,
        "llm_metrics",
        instance,
    )

    yield instance

    instance.reset()


# ==========================================================
# REGISTRO DE EVENTOS
# ==========================================================


def test_trace_records_success_event(metrics):
    with metrics.track_analysis(
        "analysis-success",
        request_id="request-success",
    ):
        metrics.record_success(
            model="gemini-test-model",
            duration_ms=120.0,
            prompt_tokens=100,
            output_tokens=50,
            thinking_tokens=10,
            total_tokens=160,
            provider="gemini",
        )

    traces = metrics.get_recent_traces()

    assert len(traces) == 1

    event = traces[0]

    assert event["event_type"] == "llm_call"
    assert event["status"] == "success"
    assert event["analysis_id"] == "analysis-success"
    assert event["request_id"] == "request-success"
    assert event["provider"] == "gemini"
    assert event["model"] == "gemini-test-model"
    assert event["duration_ms"] == 120.0

    assert event["tokens"] == {
        "input": 100,
        "output": 50,
        "thinking": 10,
        "total": 160,
    }

    UUID(event["event_id"])

    assert event["timestamp"].endswith("Z")


def test_trace_records_failure_event_with_error_type(metrics):
    with metrics.track_analysis(
        "analysis-failure",
        request_id="request-failure",
    ):
        metrics.record_failure(
            model="openai-test-model",
            duration_ms=450.0,
            provider="openai",
            error_type="LLMTimeoutError",
        )

    traces = metrics.get_recent_traces()

    assert len(traces) == 1

    event = traces[0]

    assert event["event_type"] == "llm_call"
    assert event["status"] == "failure"
    assert event["provider"] == "openai"
    assert event["model"] == "openai-test-model"
    assert event["duration_ms"] == 450.0
    assert event["error_type"] == "LLMTimeoutError"
    assert event["analysis_id"] == "analysis-failure"
    assert event["request_id"] == "request-failure"


def test_trace_records_retry_metadata(metrics):
    with metrics.track_analysis(
        "analysis-retry",
        request_id="request-retry",
    ):
        metrics.record_retry(
            provider="anthropic",
            model="claude-test-model",
            attempt=1,
            max_attempts=3,
            delay_ms=1000.0,
            error_type="LLMTransientProviderError",
        )

    traces = metrics.get_recent_traces()

    assert len(traces) == 1

    event = traces[0]

    assert event["event_type"] == "llm_retry"
    assert event["status"] == "scheduled"
    assert event["provider"] == "anthropic"
    assert event["model"] == "claude-test-model"

    assert event["retry"] == {
        "attempt": 1,
        "next_attempt": 2,
        "max_attempts": 3,
        "delay_ms": 1000.0,
    }

    assert event["error_type"] == "LLMTransientProviderError"
    assert event["analysis_id"] == "analysis-retry"
    assert event["request_id"] == "request-retry"


def test_trace_events_have_unique_ids(metrics):
    for index in range(10):
        metrics.record_success(
            model="gemini-test-model",
            duration_ms=10.0,
            provider="gemini",
        )

    traces = metrics.get_recent_traces(limit=10)

    event_ids = [event["event_id"] for event in traces]

    assert len(event_ids) == 10
    assert len(set(event_ids)) == 10

    for event_id in event_ids:
        UUID(event_id)


def test_trace_does_not_store_prompts_or_responses(metrics):
    metrics.record_success(
        model="gemini-test-model",
        duration_ms=100.0,
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    event = metrics.get_recent_traces()[0]

    assert "prompt" not in event
    assert "response" not in event
    assert "user_prompt" not in event
    assert "system_prompt" not in event
    assert "response_text" not in event


def test_trace_events_are_returned_newest_first(metrics):
    for index in range(3):
        with metrics.track_analysis(f"analysis-{index}"):
            metrics.record_success(
                model="gemini-test-model",
                duration_ms=float(index + 1),
            )

    traces = metrics.get_recent_traces()

    assert [event["analysis_id"] for event in traces] == [
        "analysis-2",
        "analysis-1",
        "analysis-0",
    ]


def test_trace_results_are_copies(metrics):
    metrics.record_success(
        model="gemini-test-model",
        duration_ms=100.0,
    )

    first_result = metrics.get_recent_traces()

    first_result[0]["provider"] = "modified"
    first_result[0]["tokens"]["input"] = 999

    second_result = metrics.get_recent_traces()

    assert second_result[0]["provider"] == "gemini"
    assert second_result[0]["tokens"]["input"] is None


# ==========================================================
# RETENÇÃO
# ==========================================================


def test_trace_buffer_respects_maximum_capacity():
    metrics = LLMMetrics(max_trace_events=2)

    for index in range(3):
        with metrics.track_analysis(f"analysis-{index}"):
            metrics.record_success(
                model="gemini-test-model",
                duration_ms=10.0,
                provider="gemini",
            )

    traces = metrics.get_recent_traces(limit=10)

    assert len(traces) == 2

    assert [event["analysis_id"] for event in traces] == [
        "analysis-2",
        "analysis-1",
    ]

    # O buffer de traces não limita os acumuladores globais.
    snapshot = metrics.snapshot("gemini")

    assert snapshot["requests"]["total"] == 3


@pytest.mark.parametrize(
    "capacity",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_trace_buffer_rejects_invalid_capacity(capacity):
    with pytest.raises((TypeError, ValueError)):
        LLMMetrics(max_trace_events=capacity)


def test_trace_query_limit_cannot_be_zero(metrics):
    metrics.record_success(
        model="gemini-test-model",
        duration_ms=100.0,
    )

    with pytest.raises(ValueError, match="limit"):
        metrics.get_recent_traces(limit=0)


def test_trace_query_limit_rejects_boolean(metrics):
    with pytest.raises(TypeError, match="limit"):
        metrics.get_recent_traces(limit=True)


# ==========================================================
# FILTROS DOS TRACES
# ==========================================================


def seed_trace_events(metrics):
    """Cria eventos de diferentes provedores e análises."""

    with metrics.track_analysis(
        "analysis-gemini",
        request_id="request-gemini",
    ):
        metrics.record_success(
            model="gemini-test-model",
            duration_ms=100.0,
            provider="gemini",
        )

        metrics.record_retry(
            provider="gemini",
            model="gemini-test-model",
            attempt=1,
            max_attempts=3,
            delay_ms=1000.0,
            error_type="LLMTransientProviderError",
        )

    with metrics.track_analysis(
        "analysis-openai",
        request_id="request-openai",
    ):
        metrics.record_failure(
            model="openai-test-model",
            duration_ms=200.0,
            provider="openai",
            error_type="LLMTimeoutError",
        )


def test_get_recent_traces_filters_by_provider(metrics):
    seed_trace_events(metrics)

    gemini_traces = metrics.get_recent_traces(
        provider="gemini",
    )

    openai_traces = metrics.get_recent_traces(
        provider="openai",
    )

    assert len(gemini_traces) == 2
    assert len(openai_traces) == 1

    assert all(event["provider"] == "gemini" for event in gemini_traces)

    assert openai_traces[0]["provider"] == "openai"


def test_get_recent_traces_filters_by_analysis_id(metrics):
    seed_trace_events(metrics)

    traces = metrics.get_recent_traces(
        analysis_id="analysis-gemini",
    )

    assert len(traces) == 2

    assert all(event["analysis_id"] == "analysis-gemini" for event in traces)


def test_get_recent_traces_filters_by_request_id(metrics):
    seed_trace_events(metrics)

    traces = metrics.get_recent_traces(
        request_id="request-openai",
    )

    assert len(traces) == 1

    assert traces[0]["request_id"] == "request-openai"
    assert traces[0]["provider"] == "openai"


def test_get_recent_traces_filters_by_status(metrics):
    seed_trace_events(metrics)

    scheduled = metrics.get_recent_traces(
        status="scheduled",
    )

    failures = metrics.get_recent_traces(
        status="failure",
    )

    successes = metrics.get_recent_traces(
        status="success",
    )

    assert len(scheduled) == 1
    assert len(failures) == 1
    assert len(successes) == 1

    assert scheduled[0]["event_type"] == "llm_retry"
    assert failures[0]["error_type"] == "LLMTimeoutError"
    assert successes[0]["provider"] == "gemini"


def test_get_recent_traces_combines_filters(metrics):
    seed_trace_events(metrics)

    traces = metrics.get_recent_traces(
        provider="gemini",
        analysis_id="analysis-gemini",
        request_id="request-gemini",
        status="success",
        limit=10,
    )

    assert len(traces) == 1
    assert traces[0]["status"] == "success"


def test_get_recent_traces_respects_limit(metrics):
    for index in range(5):
        metrics.record_success(
            model="gemini-test-model",
            duration_ms=10.0,
        )

    traces = metrics.get_recent_traces(limit=2)

    assert len(traces) == 2


def test_reset_clears_trace_events(metrics):
    metrics.record_success(
        model="gemini-test-model",
        duration_ms=100.0,
    )

    assert len(metrics.get_recent_traces()) == 1

    metrics.reset()

    assert metrics.get_recent_traces() == []


# ==========================================================
# DISTRIBUIÇÃO DE LATÊNCIA
# ==========================================================


def test_latency_distribution_calculates_percentiles(metrics):
    durations = [10.0, 20.0, 30.0, 40.0]

    with metrics.track_analysis("analysis-latency") as analysis:
        for duration in durations:
            metrics.record_success(
                model="gemini-test-model",
                duration_ms=duration,
                provider="gemini",
            )

        analysis_snapshot = analysis.snapshot()

    snapshot = metrics.snapshot("gemini")

    assert snapshot["latency"] == {
        "average_ms": 25.0,
    }

    assert snapshot["latency_distribution"] == {
        "p50_ms": 25.0,
        "p95_ms": 38.5,
        "sample_count": 4,
        "sample_capacity": 1000,
    }

    assert analysis_snapshot["latency"] == {
        "total_ms": 100.0,
        "average_ms": 25.0,
    }

    assert analysis_snapshot["latency_distribution"] == {
        "p50_ms": 25.0,
        "p95_ms": 38.5,
        "sample_count": 4,
        "sample_capacity": 1000,
    }


def test_latency_distribution_includes_failed_requests(metrics):
    metrics.record_success(
        model="openai-test-model",
        duration_ms=100.0,
        provider="openai",
    )

    metrics.record_failure(
        model="openai-test-model",
        duration_ms=300.0,
        provider="openai",
    )

    snapshot = metrics.snapshot("openai")

    assert snapshot["requests"]["total"] == 2

    assert snapshot["latency_distribution"] == {
        "p50_ms": 200.0,
        "p95_ms": 290.0,
        "sample_count": 2,
        "sample_capacity": 1000,
    }


def test_latency_distribution_does_not_count_retry_as_request(metrics):
    metrics.record_retry(
        provider="gemini",
        model="gemini-test-model",
        attempt=1,
        max_attempts=3,
        delay_ms=1000.0,
    )

    snapshot = metrics.snapshot("gemini")

    assert snapshot["requests"]["total"] == 0
    assert snapshot["requests"]["retries"] == 1

    assert snapshot["latency_distribution"] == {
        "p50_ms": 0.0,
        "p95_ms": 0.0,
        "sample_count": 0,
        "sample_capacity": 1000,
    }


# ==========================================================
# ENDPOINT DE TRACES
# ==========================================================


def test_llm_traces_endpoint_returns_recent_events(api_metrics):
    with api_metrics.track_analysis(
        "analysis-api",
        request_id="request-api",
    ):
        api_metrics.record_success(
            model="gemini-test-model",
            duration_ms=120.0,
            provider="gemini",
        )

    response = client.get("/metrics/llm/traces")

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 1
    assert data["limit"] == 100
    assert len(data["items"]) == 1

    event = data["items"][0]

    assert event["analysis_id"] == "analysis-api"
    assert event["request_id"] == "request-api"
    assert event["provider"] == "gemini"
    assert event["status"] == "success"


def test_llm_traces_endpoint_applies_all_filters(api_metrics):
    seed_trace_events(api_metrics)

    response = client.get(
        "/metrics/llm/traces",
        params={
            "provider": "gemini",
            "analysis_id": "analysis-gemini",
            "request_id": "request-gemini",
            "status": "scheduled",
            "limit": 10,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["count"] == 1

    event = data["items"][0]

    assert event["event_type"] == "llm_retry"
    assert event["provider"] == "gemini"
    assert event["analysis_id"] == "analysis-gemini"
    assert event["request_id"] == "request-gemini"
    assert event["status"] == "scheduled"


def test_llm_traces_endpoint_returns_empty_list_without_events(api_metrics):
    response = client.get("/metrics/llm/traces")

    assert response.status_code == 200

    assert response.json() == {
        "items": [],
        "count": 0,
        "limit": 100,
    }


def test_llm_traces_endpoint_rejects_invalid_status(api_metrics):
    response = client.get(
        "/metrics/llm/traces",
        params={"status": "pending"},
    )

    assert response.status_code == 422


def test_llm_traces_endpoint_rejects_invalid_provider(api_metrics):
    response = client.get(
        "/metrics/llm/traces",
        params={"provider": "invalid-provider"},
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "limit",
    [
        0,
        101,
    ],
)
def test_llm_traces_endpoint_rejects_invalid_limit(api_metrics, limit):
    response = client.get(
        "/metrics/llm/traces",
        params={"limit": limit},
    )

    assert response.status_code == 422


# ==========================================================
# CORRELAÇÃO REAL COM REQUEST ID HTTP
# ==========================================================


def test_analyze_correlates_trace_with_http_request_id(
    monkeypatch,
    tmp_path,
    api_metrics,
):
    """
    Verifica a propagação real do request_id do middleware
    até os eventos associados à análise.

    Utiliza um BusinessAnalyst simulado, sem chamadas externas.
    """

    history = AnalysisHistory(tmp_path / "tracing_history.sqlite3")

    monkeypatch.setattr(
        api_main,
        "analysis_history",
        history,
    )

    class FakeBusinessAnalyst:
        def __init__(self, cache, provider_name):
            self.model = "gemini-test-model"
            self.steps = []
            self.last_sql = None
            self.provider_name = provider_name

        def analyze(self, question):
            api_metrics.record_success(
                model=self.model,
                duration_ms=120.0,
                prompt_tokens=100,
                output_tokens=50,
                total_tokens=150,
                provider=self.provider_name,
            )

            return {
                "answer": "Análise simulada.",
                "steps": [],
                "sql": None,
                "queries": [],
                "results": [],
                "total_results": 0,
                "chart": None,
                "cache_hit": False,
            }

    monkeypatch.setattr(
        api_main,
        "BusinessAnalyst",
        FakeBusinessAnalyst,
    )

    response = client.post(
        "/analyze",
        json={
            "question": "Quantos pedidos foram entregues?",
            "provider": "gemini",
        },
    )

    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]
    analysis_id = response.json()["analysis_id"]

    UUID(request_id)
    UUID(analysis_id)

    trace_response = client.get(
        "/metrics/llm/traces",
        params={
            "request_id": request_id,
            "analysis_id": analysis_id,
        },
    )

    assert trace_response.status_code == 200

    data = trace_response.json()

    assert data["count"] == 1

    event = data["items"][0]

    assert event["request_id"] == request_id
    assert event["analysis_id"] == analysis_id
    assert event["provider"] == "gemini"
    assert event["model"] == "gemini-test-model"
    assert event["status"] == "success"
