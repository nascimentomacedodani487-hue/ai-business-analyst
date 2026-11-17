"""Precificação e estimativa de custos dos provedores de LLM.

Os preços são expressos em USD por milhão de tokens.

O estimador utiliza tarifas padrão, sem descontos de cache,
Batch API, Flex ou processamento prioritário.

Modelos antigos permanecem cadastrados para preservar
a interpretação dos relatórios históricos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

PricingTier = Literal["standard", "free"]

TOKENS_PER_MILLION = Decimal("1000000")
PRICING_AS_OF = date(2026, 11, 15)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Tarifas de um modelo específico."""

    provider: str
    model: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    source_url: str
    free_tier_available: bool = False
    output_includes_thinking_tokens: bool = False
    long_context_threshold_tokens: int | None = None
    long_context_input_usd_per_million: Decimal | None = None
    long_context_output_usd_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        """Valida a configuração das tarifas opcionais de contexto longo."""

        long_context_values = (
            self.long_context_threshold_tokens,
            self.long_context_input_usd_per_million,
            self.long_context_output_usd_per_million,
        )

        configured_values = [value is not None for value in long_context_values]

        if any(configured_values) and not all(configured_values):
            raise ValueError(
                "As três configurações de contexto longo "
                "devem ser informadas em conjunto."
            )

        if self.long_context_threshold_tokens is not None:
            if (
                isinstance(self.long_context_threshold_tokens, bool)
                or not isinstance(self.long_context_threshold_tokens, int)
                or self.long_context_threshold_tokens <= 0
            ):
                raise ValueError(
                    "long_context_threshold_tokens deve ser um inteiro positivo."
                )

            if (
                self.long_context_input_usd_per_million < 0
                or self.long_context_output_usd_per_million < 0
            ):
                raise ValueError(
                    "As tarifas de contexto longo não podem ser negativas."
                )


MODEL_PRICING: dict[tuple[str, str], ModelPricing] = {
    # ======================================================
    # GOOGLE GEMINI
    # ======================================================
    (
        "gemini",
        "gemini-2.5-flash",
    ): ModelPricing(
        provider="gemini",
        model="gemini-2.5-flash",
        input_usd_per_million=Decimal("0.30"),
        output_usd_per_million=Decimal("2.50"),
        source_url="https://ai.google.dev/gemini-api/docs/pricing",
        free_tier_available=True,
        output_includes_thinking_tokens=True,
    ),
    # ======================================================
    # OPENAI
    # ======================================================
    # Mantido para compatibilidade com relatórios anteriores.
    (
        "openai",
        "gpt-4.1-mini",
    ): ModelPricing(
        provider="openai",
        model="gpt-4.1-mini",
        input_usd_per_million=Decimal("0.40"),
        output_usd_per_million=Decimal("1.60"),
        source_url="https://platform.openai.com/pricing",
    ),
    # Modelo considerado para a próxima integração OpenAI.
    (
        "openai",
        "gpt-6-luna",
    ): ModelPricing(
        provider="openai",
        model="gpt-6-luna",
        input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("0.50"),
        source_url="https://developers.openai.com/api/docs/models/gpt-6-luna",
        long_context_threshold_tokens=272_000,
        long_context_input_usd_per_million=Decimal("0.20"),
        long_context_output_usd_per_million=Decimal("0.75"),
    ),
    # ======================================================
    # ANTHROPIC
    # ======================================================
    # Mantido para compatibilidade com relatórios anteriores.
    (
        "anthropic",
        "claude-haiku-4-5-20251001",
    ): ModelPricing(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        input_usd_per_million=Decimal("1.00"),
        output_usd_per_million=Decimal("5.00"),
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
    ),
    # Modelo considerado para a próxima integração Anthropic.
    (
        "anthropic",
        "claude-haiku-5-5",
    ): ModelPricing(
        provider="anthropic",
        model="claude-haiku-5-5",
        input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("0.50"),
        source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        long_context_threshold_tokens=100_000,
        long_context_input_usd_per_million=Decimal("0.50"),
        long_context_output_usd_per_million=Decimal("2.50"),
    ),
}


def _validate_token_count(
    value: int,
    field_name: str,
) -> int:
    """Valida um contador de tokens."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} deve ser um inteiro.")

    if value < 0:
        raise ValueError(f"{field_name} não pode ser negativo.")

    return value


def _to_float(value: Decimal) -> float:
    """Converte valores financeiros preservando precisão útil."""

    return float(
        value.quantize(
            Decimal("0.000000000001"),
            rounding=ROUND_HALF_UP,
        )
    )


def get_model_pricing(
    provider: str,
    model: str,
) -> ModelPricing | None:
    """Obtém as tarifas cadastradas para um modelo."""

    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()

    return MODEL_PRICING.get((normalized_provider, normalized_model))


def estimate_token_cost(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    thinking_tokens: int | None = None,
    pricing_tier: PricingTier = "standard",
) -> dict[str, object] | None:
    """Estima o custo de uma execução a partir do consumo de tokens.

    Retorna None quando o modelo não possui tarifas cadastradas.

    Regras especiais:

    - Gemini: tokens de raciocínio são incluídos na saída faturável.
    - OpenAI GPT-6 Luna: utiliza tarifas maiores acima de 272 mil
      tokens de entrada.
    - Anthropic Haiku 5.5: utiliza tarifas maiores acima de 100 mil
      tokens de entrada.
    - OpenAI e demais modelos Anthropic: utiliza os contadores de
      saída fornecidos pelos respectivos adaptadores.
    - O nível gratuito precisa ser explicitamente informado.

    O cálculo considera tarifas padrão, sem cache de prompt,
    Batch API, Flex ou adicionais de ferramentas.
    """

    if pricing_tier not in ("standard", "free"):
        raise ValueError("pricing_tier deve ser 'standard' ou 'free'.")

    normalized_input = _validate_token_count(
        input_tokens,
        "input_tokens",
    )

    normalized_output = _validate_token_count(
        output_tokens,
        "output_tokens",
    )

    normalized_thinking = (
        _validate_token_count(
            thinking_tokens,
            "thinking_tokens",
        )
        if thinking_tokens is not None
        else 0
    )

    pricing = get_model_pricing(
        provider,
        model,
    )

    if pricing is None:
        return None

    if pricing_tier == "free" and not pricing.free_tier_available:
        raise ValueError(f"O modelo {model} não possui nível gratuito cadastrado.")

    # ------------------------------------------------------
    # Seleção da faixa de contexto.
    # ------------------------------------------------------

    long_context_applies = (
        pricing.long_context_threshold_tokens is not None
        and normalized_input > pricing.long_context_threshold_tokens
    )

    if long_context_applies:
        selected_input_rate = pricing.long_context_input_usd_per_million

        selected_output_rate = pricing.long_context_output_usd_per_million

        context_tier = "long_context"

    else:
        selected_input_rate = pricing.input_usd_per_million
        selected_output_rate = pricing.output_usd_per_million

        context_tier = "standard_context"

    # ------------------------------------------------------
    # Cálculo dos tokens faturáveis.
    # ------------------------------------------------------

    billed_output_tokens = normalized_output + (
        normalized_thinking if pricing.output_includes_thinking_tokens else 0
    )

    # ------------------------------------------------------
    # Cálculo financeiro.
    # ------------------------------------------------------

    if pricing_tier == "free":
        effective_input_rate = Decimal("0")
        effective_output_rate = Decimal("0")

    else:
        effective_input_rate = selected_input_rate
        effective_output_rate = selected_output_rate

    input_cost = Decimal(normalized_input) * effective_input_rate / TOKENS_PER_MILLION

    output_cost = (
        Decimal(billed_output_tokens) * effective_output_rate / TOKENS_PER_MILLION
    )

    total_cost = input_cost + output_cost

    return {
        "provider": pricing.provider,
        "model": pricing.model,
        "pricing_tier": pricing_tier,
        "pricing_context_tier": context_tier,
        "currency": "USD",
        "pricing_as_of": PRICING_AS_OF.isoformat(),
        "pricing_source": pricing.source_url,
        "long_context_threshold_tokens": (pricing.long_context_threshold_tokens),
        "tokens": {
            "input": normalized_input,
            "output": normalized_output,
            "thinking": normalized_thinking,
            "billable_output": billed_output_tokens,
        },
        "rates": {
            "input_usd_per_million": float(effective_input_rate),
            "output_usd_per_million": float(effective_output_rate),
        },
        "input_cost_usd": _to_float(input_cost),
        "output_cost_usd": _to_float(output_cost),
        "estimated_cost_usd": _to_float(total_cost),
    }
