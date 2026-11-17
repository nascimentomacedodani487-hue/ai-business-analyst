"""Testes da precificação dos provedores de LLM."""

from decimal import Decimal

import pytest

from ai_analyst.llm.pricing import (
    MODEL_PRICING,
    PRICING_AS_OF,
    ModelPricing,
    estimate_token_cost,
    get_model_pricing,
)

# ==========================================================
# CATÁLOGO DE MODELOS
# ==========================================================


def test_model_pricing_contains_configured_models():
    assert ("gemini", "gemini-2.5-flash") in MODEL_PRICING

    assert ("openai", "gpt-4.1-mini") in MODEL_PRICING
    assert ("openai", "gpt-6-luna") in MODEL_PRICING

    assert (
        "anthropic",
        "claude-haiku-4-5-20251001",
    ) in MODEL_PRICING

    assert (
        "anthropic",
        "claude-haiku-5-5",
    ) in MODEL_PRICING


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("gemini", "gemini-2.5-flash"),
        ("openai", "gpt-4.1-mini"),
        ("openai", "gpt-6-luna"),
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("anthropic", "claude-haiku-5-5"),
    ],
)
def test_get_model_pricing_returns_registered_model(
    provider,
    model,
):
    pricing = get_model_pricing(provider, model)

    assert pricing is not None
    assert pricing.provider == provider
    assert pricing.model == model


def test_get_model_pricing_returns_none_for_unknown_model():
    assert (
        get_model_pricing(
            "openai",
            "unknown-model",
        )
        is None
    )


def test_get_model_pricing_normalizes_provider():
    pricing = get_model_pricing(
        "OPENAI",
        "gpt-4.1-mini",
    )

    assert pricing is not None
    assert pricing.provider == "openai"


# ==========================================================
# VALIDAÇÃO DA CONFIGURAÇÃO
# ==========================================================


def test_model_pricing_rejects_incomplete_long_context_configuration():
    with pytest.raises(
        ValueError,
        match="devem ser informadas em conjunto",
    ):
        ModelPricing(
            provider="test",
            model="test-model",
            input_usd_per_million=Decimal("1"),
            output_usd_per_million=Decimal("2"),
            source_url="https://example.com",
            long_context_threshold_tokens=100_000,
        )


def test_model_pricing_rejects_invalid_long_context_threshold():
    with pytest.raises(
        ValueError,
        match="inteiro positivo",
    ):
        ModelPricing(
            provider="test",
            model="test-model",
            input_usd_per_million=Decimal("1"),
            output_usd_per_million=Decimal("2"),
            source_url="https://example.com",
            long_context_threshold_tokens=True,
            long_context_input_usd_per_million=Decimal("2"),
            long_context_output_usd_per_million=Decimal("3"),
        )


# ==========================================================
# GEMINI
# ==========================================================


def test_estimate_gemini_standard_cost_includes_thinking_tokens():
    result = estimate_token_cost(
        provider="gemini",
        model="gemini-2.5-flash",
        input_tokens=1_000_000,
        output_tokens=100_000,
        thinking_tokens=20_000,
    )

    assert result is not None
    assert result["pricing_tier"] == "standard"
    assert result["pricing_context_tier"] == "standard_context"
    assert result["currency"] == "USD"

    assert result["tokens"] == {
        "input": 1_000_000,
        "output": 100_000,
        "thinking": 20_000,
        "billable_output": 120_000,
    }

    assert result["input_cost_usd"] == pytest.approx(0.30)
    assert result["output_cost_usd"] == pytest.approx(0.30)
    assert result["estimated_cost_usd"] == pytest.approx(0.60)


def test_estimate_gemini_free_tier_returns_zero_cost():
    result = estimate_token_cost(
        provider="gemini",
        model="gemini-2.5-flash",
        input_tokens=500_000,
        output_tokens=200_000,
        thinking_tokens=50_000,
        pricing_tier="free",
    )

    assert result is not None
    assert result["pricing_tier"] == "free"

    assert result["input_cost_usd"] == 0.0
    assert result["output_cost_usd"] == 0.0
    assert result["estimated_cost_usd"] == 0.0

    assert result["rates"] == {
        "input_usd_per_million": 0.0,
        "output_usd_per_million": 0.0,
    }


# ==========================================================
# OPENAI
# ==========================================================


def test_estimate_openai_legacy_model_cost_without_double_counting():
    result = estimate_token_cost(
        provider="openai",
        model="gpt-4.1-mini",
        input_tokens=1_000_000,
        output_tokens=500_000,
        thinking_tokens=100_000,
    )

    assert result is not None

    assert result["tokens"]["billable_output"] == 500_000

    assert result["input_cost_usd"] == pytest.approx(0.40)
    assert result["output_cost_usd"] == pytest.approx(0.80)
    assert result["estimated_cost_usd"] == pytest.approx(1.20)


def test_estimate_gpt6_luna_standard_context_cost():
    """Uma entrada abaixo de 272 mil tokens utiliza a tarifa padrão."""

    result = estimate_token_cost(
        provider="openai",
        model="gpt-6-luna",
        input_tokens=200_000,
        output_tokens=500_000,
    )

    assert result is not None
    assert result["pricing_context_tier"] == "standard_context"

    assert result["rates"]["input_usd_per_million"] == pytest.approx(0.10)
    assert result["rates"]["output_usd_per_million"] == pytest.approx(0.50)

    assert result["input_cost_usd"] == pytest.approx(0.02)
    assert result["output_cost_usd"] == pytest.approx(0.25)
    assert result["estimated_cost_usd"] == pytest.approx(0.27)


def test_estimate_gpt6_luna_long_context_cost():
    """Uma entrada acima de 272 mil tokens utiliza a tarifa ampliada."""

    result = estimate_token_cost(
        provider="openai",
        model="gpt-6-luna",
        input_tokens=273_000,
        output_tokens=100_000,
    )

    assert result is not None
    assert result["pricing_context_tier"] == "long_context"
    assert result["long_context_threshold_tokens"] == 272_000

    assert result["rates"]["input_usd_per_million"] == pytest.approx(0.20)
    assert result["rates"]["output_usd_per_million"] == pytest.approx(0.75)

    assert result["input_cost_usd"] == pytest.approx(0.0546)
    assert result["output_cost_usd"] == pytest.approx(0.075)
    assert result["estimated_cost_usd"] == pytest.approx(0.1296)


def test_estimate_gpt6_luna_threshold_uses_standard_rate_at_boundary():
    """O limite exato de 272 mil ainda utiliza a tarifa padrão."""

    result = estimate_token_cost(
        provider="openai",
        model="gpt-6-luna",
        input_tokens=272_000,
        output_tokens=100_000,
    )

    assert result is not None
    assert result["pricing_context_tier"] == "standard_context"

    assert result["rates"]["input_usd_per_million"] == pytest.approx(0.10)
    assert result["rates"]["output_usd_per_million"] == pytest.approx(0.50)


# ==========================================================
# ANTHROPIC
# ==========================================================


def test_estimate_anthropic_legacy_haiku_45_cost():
    result = estimate_token_cost(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=200_000,
    )

    assert result is not None

    assert result["input_cost_usd"] == pytest.approx(1.00)
    assert result["output_cost_usd"] == pytest.approx(1.00)
    assert result["estimated_cost_usd"] == pytest.approx(2.00)


def test_estimate_haiku_55_standard_context_cost():
    result = estimate_token_cost(
        provider="anthropic",
        model="claude-haiku-5-5",
        input_tokens=100_000,
        output_tokens=20_000,
    )

    assert result is not None
    assert result["pricing_context_tier"] == "standard_context"

    assert result["rates"]["input_usd_per_million"] == pytest.approx(0.10)
    assert result["rates"]["output_usd_per_million"] == pytest.approx(0.50)

    assert result["input_cost_usd"] == pytest.approx(0.01)
    assert result["output_cost_usd"] == pytest.approx(0.01)
    assert result["estimated_cost_usd"] == pytest.approx(0.02)


def test_estimate_haiku_55_long_context_cost():
    result = estimate_token_cost(
        provider="anthropic",
        model="claude-haiku-5-5",
        input_tokens=100_001,
        output_tokens=20_000,
    )

    assert result is not None
    assert result["pricing_context_tier"] == "long_context"
    assert result["long_context_threshold_tokens"] == 100_000

    assert result["rates"]["input_usd_per_million"] == pytest.approx(0.50)
    assert result["rates"]["output_usd_per_million"] == pytest.approx(2.50)

    assert result["input_cost_usd"] == pytest.approx(0.0500005)
    assert result["output_cost_usd"] == pytest.approx(0.05)
    assert result["estimated_cost_usd"] == pytest.approx(0.1000005)


def test_estimate_haiku_55_threshold_uses_standard_rate_at_boundary():
    """O limite exato de 100 mil ainda utiliza a tarifa padrão."""

    result = estimate_token_cost(
        provider="anthropic",
        model="claude-haiku-5-5",
        input_tokens=100_000,
        output_tokens=20_000,
    )

    assert result is not None
    assert result["pricing_context_tier"] == "standard_context"

    assert result["rates"]["input_usd_per_million"] == pytest.approx(0.10)
    assert result["rates"]["output_usd_per_million"] == pytest.approx(0.50)


# ==========================================================
# MODELOS DESCONHECIDOS
# ==========================================================


def test_estimate_returns_none_for_unknown_model():
    result = estimate_token_cost(
        provider="openai",
        model="unknown-model",
        input_tokens=100,
        output_tokens=50,
    )

    assert result is None


# ==========================================================
# VALIDAÇÃO DE TOKENS
# ==========================================================


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("input_tokens", -1, ValueError),
        ("output_tokens", -1, ValueError),
        ("input_tokens", True, TypeError),
        ("output_tokens", 1.5, TypeError),
        ("thinking_tokens", -1, ValueError),
        ("thinking_tokens", False, TypeError),
    ],
)
def test_estimate_rejects_invalid_token_counts(
    field,
    value,
    exception,
):
    arguments = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "input_tokens": 100,
        "output_tokens": 50,
    }

    arguments[field] = value

    with pytest.raises(exception):
        estimate_token_cost(**arguments)


# ==========================================================
# VALIDAÇÃO DE TARIFAS
# ==========================================================


def test_estimate_rejects_invalid_pricing_tier():
    with pytest.raises(
        ValueError,
        match="pricing_tier",
    ):
        estimate_token_cost(
            provider="gemini",
            model="gemini-2.5-flash",
            input_tokens=100,
            output_tokens=50,
            pricing_tier="invalid",
        )


def test_estimate_rejects_free_tier_for_unsupported_provider():
    with pytest.raises(
        ValueError,
        match="nível gratuito",
    ):
        estimate_token_cost(
            provider="openai",
            model="gpt-4.1-mini",
            input_tokens=100,
            output_tokens=50,
            pricing_tier="free",
        )


def test_estimate_zero_tokens_returns_zero_cost():
    result = estimate_token_cost(
        provider="openai",
        model="gpt-4.1-mini",
        input_tokens=0,
        output_tokens=0,
    )

    assert result is not None
    assert result["estimated_cost_usd"] == 0.0


# ==========================================================
# METADADOS
# ==========================================================


def test_estimate_includes_pricing_metadata():
    result = estimate_token_cost(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        input_tokens=1_000,
        output_tokens=500,
    )

    assert result is not None
    assert result["pricing_as_of"] == PRICING_AS_OF.isoformat()
    assert result["pricing_source"].startswith("https://")

    assert result["rates"]["input_usd_per_million"] == 1.0
    assert result["rates"]["output_usd_per_million"] == 5.0


def test_estimate_includes_gpt6_luna_pricing_metadata():
    result = estimate_token_cost(
        provider="openai",
        model="gpt-6-luna",
        input_tokens=1_000,
        output_tokens=500,
    )

    assert result is not None
    assert result["model"] == "gpt-6-luna"
    assert result["pricing_context_tier"] == "standard_context"
    assert result["pricing_source"].startswith("https://developers.openai.com/")


def test_estimate_includes_haiku_55_pricing_metadata():
    result = estimate_token_cost(
        provider="anthropic",
        model="claude-haiku-5-5",
        input_tokens=1_000,
        output_tokens=500,
    )

    assert result is not None
    assert result["model"] == "claude-haiku-5-5"
    assert result["pricing_context_tier"] == "standard_context"
    assert result["pricing_source"].startswith("https://platform.claude.com/")
