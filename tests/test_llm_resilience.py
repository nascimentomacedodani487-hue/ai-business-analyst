import pytest

from ai_analyst.llm.base import (
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTransientProviderError,
)
from ai_analyst.llm.resilience import RetryLLMProvider


class FakeProvider:
    """Provedor simulado com respostas e erros configuráveis."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_response(
        self,
        user_prompt,
        system_prompt,
        max_output_tokens=512,
    ):
        self.calls += 1

        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


def test_retry_recovers_after_timeout():
    provider = FakeProvider(
        [
            LLMTimeoutError("Timeout"),
            "Resposta recuperada.",
        ]
    )

    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=3,
        initial_delay=1,
        sleep_fn=delays.append,
    )

    response = retry.generate_response(
        "Pergunta",
        "Instruções",
    )

    assert response == "Resposta recuperada."
    assert provider.calls == 2
    assert delays == [1]


def test_retry_recovers_after_transient_provider_error():
    provider = FakeProvider(
        [
            LLMTransientProviderError("Servidor indisponível"),
            LLMTransientProviderError("Servidor indisponível"),
            "Resposta recuperada.",
        ]
    )

    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=3,
        initial_delay=1,
        sleep_fn=delays.append,
    )

    response = retry.generate_response(
        "Pergunta",
        "Instruções",
    )

    assert response == "Resposta recuperada."
    assert provider.calls == 3
    assert delays == [1, 2]


def test_retry_uses_exponential_backoff():
    provider = FakeProvider(
        [
            LLMTimeoutError("Timeout 1"),
            LLMTimeoutError("Timeout 2"),
            "Sucesso",
        ]
    )

    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=3,
        initial_delay=2,
        sleep_fn=delays.append,
    )

    response = retry.generate_response(
        "Pergunta",
        "Instruções",
    )

    assert response == "Sucesso"
    assert delays == [2, 4]


def test_retry_stops_after_max_attempts():
    error = LLMTimeoutError("Timeout persistente")

    provider = FakeProvider(
        [
            error,
            error,
            error,
        ]
    )

    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=3,
        initial_delay=1,
        sleep_fn=delays.append,
    )

    with pytest.raises(LLMTimeoutError) as captured:
        retry.generate_response(
            "Pergunta",
            "Instruções",
        )

    assert captured.value is error
    assert provider.calls == 3
    assert delays == [1, 2]


@pytest.mark.parametrize(
    "error",
    [
        LLMRateLimitError("Cota esgotada"),
        LLMResponseError("Resposta inválida"),
        LLMProviderError("Erro permanente"),
        ValueError("Erro de configuração"),
    ],
)
def test_retry_does_not_repeat_permanent_errors(error):
    provider = FakeProvider([error])

    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=3,
        initial_delay=1,
        sleep_fn=delays.append,
    )

    with pytest.raises(type(error)) as captured:
        retry.generate_response(
            "Pergunta",
            "Instruções",
        )

    assert captured.value is error
    assert provider.calls == 1
    assert delays == []


def test_retry_passes_max_output_tokens():
    class TokenProvider:
        def __init__(self):
            self.received_tokens = None

        def generate_response(
            self,
            user_prompt,
            system_prompt,
            max_output_tokens=512,
        ):
            self.received_tokens = max_output_tokens
            return "Resposta"

    provider = TokenProvider()

    retry = RetryLLMProvider(
        provider=provider,
        sleep_fn=lambda _: None,
    )

    response = retry.generate_response(
        "Pergunta",
        "Instruções",
        max_output_tokens=256,
    )

    assert response == "Resposta"
    assert provider.received_tokens == 256


def test_retry_rejects_invalid_max_attempts():
    with pytest.raises(
        ValueError,
        match="max_attempts",
    ):
        RetryLLMProvider(
            provider=FakeProvider(["Resposta"]),
            max_attempts=0,
        )


def test_retry_rejects_negative_initial_delay():
    with pytest.raises(
        ValueError,
        match="initial_delay",
    ):
        RetryLLMProvider(
            provider=FakeProvider(["Resposta"]),
            initial_delay=-1,
        )


def test_retry_can_be_disabled_with_one_attempt():
    provider = FakeProvider(
        [
            LLMTimeoutError("Timeout"),
        ]
    )

    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=1,
        initial_delay=1,
        sleep_fn=delays.append,
    )

    with pytest.raises(LLMTimeoutError):
        retry.generate_response(
            "Pergunta",
            "Instruções",
        )

    assert provider.calls == 1
    assert delays == []


def test_retry_passes_structured_json_response():
    """Verifica o encaminhamento da geração JSON estruturada."""

    class StructuredProvider:
        def __init__(self):
            self.calls = 0
            self.received_schema = None
            self.received_tokens = None

        def generate_json_response(
            self,
            user_prompt,
            system_prompt,
            response_schema,
            max_output_tokens=2048,
        ):
            self.calls += 1
            self.received_schema = response_schema
            self.received_tokens = max_output_tokens

            return '{"objectives": []}'

    provider = StructuredProvider()

    retry = RetryLLMProvider(
        provider=provider,
        sleep_fn=lambda _: None,
    )

    schema = {"type": "object", "properties": {"objectives": {"type": "array"}}}

    response = retry.generate_json_response(
        user_prompt="Planeje a análise.",
        system_prompt="Você é um planejador.",
        response_schema=schema,
        max_output_tokens=2048,
    )

    assert response == '{"objectives": []}'
    assert provider.calls == 1
    assert provider.received_schema == schema
    assert provider.received_tokens == 2048


def test_retry_recovers_structured_json_after_timeout():
    """Verifica retry em uma chamada JSON estruturada."""

    class StructuredProvider:
        def __init__(self):
            self.outcomes = [
                LLMTimeoutError("Timeout"),
                '{"objectives": []}',
            ]
            self.calls = 0

        def generate_json_response(
            self,
            user_prompt,
            system_prompt,
            response_schema,
            max_output_tokens=2048,
        ):
            self.calls += 1
            outcome = self.outcomes.pop(0)

            if isinstance(outcome, Exception):
                raise outcome

            return outcome

    provider = StructuredProvider()
    delays = []

    retry = RetryLLMProvider(
        provider=provider,
        max_attempts=3,
        initial_delay=1,
        sleep_fn=delays.append,
    )

    response = retry.generate_json_response(
        user_prompt="Planeje a análise.",
        system_prompt="Você é um planejador.",
        response_schema={"type": "object"},
    )

    assert response == '{"objectives": []}'
    assert provider.calls == 2
    assert delays == [1]


def test_retry_falls_back_to_standard_response():
    """Verifica compatibilidade com provedores sem geração JSON própria."""

    provider = FakeProvider(["Resposta compatível."])

    retry = RetryLLMProvider(
        provider=provider,
        sleep_fn=lambda _: None,
    )

    response = retry.generate_json_response(
        user_prompt="Pergunta",
        system_prompt="Instruções",
        response_schema={"type": "object"},
    )

    assert response == "Resposta compatível."
    assert provider.calls == 1
