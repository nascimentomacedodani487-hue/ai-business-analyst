import pytest

from ai_analyst.llm.metrics import LLMMetrics


@pytest.fixture
def metrics():
    return LLMMetrics()


def test_metrics_initial_state(metrics):
    result = metrics.snapshot()

    assert result["provider"] == "gemini"
    assert result["pricing_tier"] == "free"
    assert result["cost_estimate_usd"] == 0.0

    assert result["requests"]["total"] == 0
    assert result["requests"]["successful"] == 0
    assert result["requests"]["failed"] == 0
    assert result["requests"]["retries"] == 0

    assert result["tokens"]["input"] == 0
    assert result["tokens"]["output"] == 0
    assert result["tokens"]["thinking"] == 0
    assert result["tokens"]["total"] == 0


def test_record_success(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=120.5,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=10,
        total_tokens=160,
    )

    result = metrics.snapshot()

    assert result["requests"]["total"] == 1
    assert result["requests"]["successful"] == 1
    assert result["requests"]["failed"] == 0

    assert result["tokens"]["input"] == 100
    assert result["tokens"]["output"] == 50
    assert result["tokens"]["thinking"] == 10
    assert result["tokens"]["total"] == 160

    assert result["latency"]["average_ms"] == 120.5


def test_record_failure(metrics):
    metrics.record_failure(
        model="gemini-2.5-flash",
        duration_ms=250.0,
    )

    result = metrics.snapshot()

    assert result["requests"]["total"] == 1
    assert result["requests"]["successful"] == 0
    assert result["requests"]["failed"] == 1
    assert result["tokens"]["unreported"] == 1
    assert result["latency"]["average_ms"] == 250.0


def test_record_retry(metrics):
    metrics.record_retry()
    metrics.record_retry()

    result = metrics.snapshot()

    assert result["requests"]["retries"] == 2
    assert result["requests"]["total"] == 0


def test_multiple_requests_aggregate_metrics(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=100.0,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=0,
        total_tokens=150,
    )

    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=300.0,
        prompt_tokens=200,
        output_tokens=100,
        thinking_tokens=20,
        total_tokens=320,
    )

    metrics.record_failure(
        model="gemini-2.5-flash",
        duration_ms=200.0,
    )

    result = metrics.snapshot()

    assert result["requests"]["total"] == 3
    assert result["requests"]["successful"] == 2
    assert result["requests"]["failed"] == 1

    assert result["tokens"]["input"] == 300
    assert result["tokens"]["output"] == 150
    assert result["tokens"]["thinking"] == 20
    assert result["tokens"]["total"] == 470

    assert result["latency"]["average_ms"] == 200.0


def test_missing_token_usage_is_not_invented(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=100.0,
    )

    result = metrics.snapshot()

    assert result["tokens"]["input"] == 0
    assert result["tokens"]["output"] == 0
    assert result["tokens"]["total"] == 0
    assert result["tokens"]["unreported"] == 1


def test_total_tokens_fallback(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=100.0,
        prompt_tokens=100,
        output_tokens=50,
        thinking_tokens=10,
    )

    result = metrics.snapshot()

    assert result["tokens"]["total"] == 160


def test_reset_clears_metrics(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=100.0,
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    metrics.record_retry()

    metrics.reset()

    result = metrics.snapshot()

    assert result["requests"]["total"] == 0
    assert result["requests"]["successful"] == 0
    assert result["requests"]["failed"] == 0
    assert result["requests"]["retries"] == 0
    assert result["tokens"]["total"] == 0
    assert result["latency"]["average_ms"] == 0.0


def test_free_tier_does_not_report_paid_cost(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=100.0,
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
    )

    result = metrics.snapshot()

    assert result["pricing_tier"] == "free"
    assert result["cost_estimate_usd"] == 0.0


def test_record_success_with_openai_provider(metrics):
    metrics.record_success(
        model="gpt-6-luna",
        duration_ms=120.0,
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="openai",
    )

    result = metrics.snapshot(provider="openai")

    assert result["provider"] == "openai"
    assert result["model"] == "gpt-6-luna"
    assert result["pricing_tier"] == "usage_based"
    assert result["cost_estimate_usd"] is None
    assert result["requests"]["total"] == 1
    assert result["tokens"]["total"] == 150


def test_record_success_with_anthropic_provider(metrics):
    metrics.record_success(
        model="claude-haiku-4-5-20251001",
        duration_ms=150.0,
        prompt_tokens=200,
        output_tokens=100,
        total_tokens=300,
        provider="anthropic",
    )

    result = metrics.snapshot(provider="anthropic")

    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-haiku-4-5-20251001"
    assert result["pricing_tier"] == "usage_based"
    assert result["cost_estimate_usd"] is None
    assert result["requests"]["successful"] == 1
    assert result["tokens"]["total"] == 300


def test_providers_have_independent_metrics(metrics):
    metrics.record_success(
        model="gemini-2.5-flash",
        duration_ms=100.0,
        prompt_tokens=100,
        output_tokens=50,
        total_tokens=150,
        provider="gemini",
    )

    metrics.record_success(
        model="gpt-6-luna",
        duration_ms=200.0,
        prompt_tokens=200,
        output_tokens=100,
        total_tokens=300,
        provider="openai",
    )

    gemini = metrics.snapshot(provider="gemini")
    openai = metrics.snapshot(provider="openai")

    assert gemini["requests"]["total"] == 1
    assert gemini["tokens"]["total"] == 150

    assert openai["requests"]["total"] == 1
    assert openai["tokens"]["total"] == 300


def test_analysis_metrics_preserve_provider(metrics):
    with metrics.track_analysis("analysis-openai") as analysis:
        metrics.record_success(
            model="gpt-6-luna",
            duration_ms=100.0,
            prompt_tokens=100,
            output_tokens=50,
            total_tokens=150,
            provider="openai",
        )

    result = analysis.snapshot()

    assert result["analysis_id"] == "analysis-openai"
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-6-luna"
    assert result["requests"]["successful"] == 1
