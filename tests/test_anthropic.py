"""Testes da integração com o provedor Anthropic."""

from types import SimpleNamespace

import pytest

import ai_analyst.llm.anthropic as anthropic_module
from ai_analyst.llm.anthropic import AnthropicClient
from ai_analyst.llm.base import (
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTransientProviderError,
)
from ai_analyst.llm.metrics import LLMMetrics


class FakeMessages:
    """Simula o endpoint messages.create da Anthropic."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs

        if self.error is not None:
            raise self.error

        return self.response


class FakeAPIStatusError(Exception):
    """Simula uma exceção HTTP do SDK da Anthropic."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.fixture
def metrics(monkeypatch):
    """Fornece uma instância isolada de métricas para cada teste."""

    instance = LLMMetrics()
    monkeypatch.setattr(anthropic_module, "llm_metrics", instance)
    return instance


def build_response(
    text="Resposta simulada do Claude.",
    input_tokens=17,
    output_tokens=9,
):
    """Cria uma resposta simulada da API."""

    content = []

    if text is not None:
        content.append(
            SimpleNamespace(
                type="text",
                text=text,
            )
        )

    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return SimpleNamespace(
        content=content,
        usage=usage,
    )


def build_client(monkeypatch, messages, api_key="test-anthropic-key", model=None):
    """Cria o cliente utilizando um SDK simulado."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)

    def fake_anthropic(*, api_key):
        assert api_key == "test-anthropic-key"
        return SimpleNamespace(messages=messages)

    monkeypatch.setattr(
        anthropic_module,
        "Anthropic",
        fake_anthropic,
    )

    if model is not None:
        return AnthropicClient(model=model)

    return AnthropicClient()


def get_only_trace(metrics):
    """Retorna o único evento registrado pelo provedor Anthropic."""

    traces = metrics.get_recent_traces(
        provider="anthropic",
        limit=10,
    )

    assert len(traces) == 1
    return traces[0]


def test_anthropic_client_initialization_without_api_key(monkeypatch):
    """Impede inicialização quando a chave não está configurada."""

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_anthropic_client_uses_default_model(monkeypatch):
    """Verifica o modelo padrão do cliente."""

    messages = FakeMessages(
        response=build_response(),
    )

    client = build_client(monkeypatch, messages)

    assert client.model == "claude-haiku-4-5-20251001"


def test_anthropic_client_accepts_custom_model(monkeypatch):
    """Permite configurar outro modelo Anthropic."""

    messages = FakeMessages(
        response=build_response(),
    )

    client = build_client(
        monkeypatch,
        messages,
        model="claude-custom-model",
    )

    assert client.model == "claude-custom-model"


def test_anthropic_success_records_tokens_and_trace(
    monkeypatch,
    metrics,
):
    """Registra tokens, duração e correlação em uma chamada bem-sucedida."""

    messages = FakeMessages(
        response=build_response(
            text="Resposta simulada do Claude.",
            input_tokens=17,
            output_tokens=9,
        ),
    )

    client = build_client(monkeypatch, messages)

    recorded_metrics = {}
    original_record_success = metrics.record_success

    def record_success(**kwargs):
        recorded_metrics.update(kwargs)
        return original_record_success(**kwargs)

    monkeypatch.setattr(
        metrics,
        "record_success",
        record_success,
    )

    with metrics.track_analysis(
        "analysis-anthropic-001",
        request_id="request-anthropic-001",
    ):
        result = client.generate_response(
            user_prompt="Analise os pedidos.",
            system_prompt="Você é um analista de negócios.",
            max_output_tokens=256,
        )

    assert result == "Resposta simulada do Claude."

    assert messages.last_request["model"] == "claude-haiku-4-5-20251001"
    assert messages.last_request["max_tokens"] == 256
    assert messages.last_request["temperature"] == 0.1
    assert messages.last_request["system"] == "Você é um analista de negócios."
    assert messages.last_request["messages"] == [
        {
            "role": "user",
            "content": "Analise os pedidos.",
        }
    ]

    assert recorded_metrics["provider"] == "anthropic"
    assert recorded_metrics["model"] == "claude-haiku-4-5-20251001"
    assert recorded_metrics["prompt_tokens"] == 17
    assert recorded_metrics["output_tokens"] == 9
    assert recorded_metrics["total_tokens"] == 26
    assert recorded_metrics["duration_ms"] >= 0

    trace = get_only_trace(metrics)

    assert trace["provider"] == "anthropic"
    assert trace["analysis_id"] == "analysis-anthropic-001"
    assert trace["request_id"] == "request-anthropic-001"


def test_anthropic_joins_multiple_text_blocks(monkeypatch, metrics):
    """Combina blocos de texto e ignora blocos que não sejam textuais."""

    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Primeira parte."),
            SimpleNamespace(type="tool_use", text=None),
            SimpleNamespace(type="text", text="Segunda parte."),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=12,
        ),
    )

    messages = FakeMessages(response=response)
    client = build_client(monkeypatch, messages)

    result = client.generate_response(
        user_prompt="Pergunta de teste.",
        system_prompt="Instruções de teste.",
    )

    assert result == "Primeira parte.\nSegunda parte."


def test_anthropic_empty_response_records_failure(monkeypatch, metrics):
    """Registra falha quando o Claude não retorna conteúdo textual."""

    messages = FakeMessages(
        response=build_response(text=None),
    )

    client = build_client(monkeypatch, messages)

    with pytest.raises(LLMResponseError, match="resposta vazia"):
        client.generate_response(
            user_prompt="Pergunta de teste.",
            system_prompt="Instruções de teste.",
        )

    trace = get_only_trace(metrics)

    assert trace["provider"] == "anthropic"
    assert trace["status"] == "failure"
    assert trace["error_type"] == "LLMResponseError"


def test_anthropic_records_original_exception_type(monkeypatch, metrics):
    """Preserva no trace o tipo da exceção original."""

    messages = FakeMessages(
        error=RuntimeError("Falha simulada de conexão."),
    )

    client = build_client(monkeypatch, messages)

    with pytest.raises(LLMProviderError):
        client.generate_response(
            user_prompt="Pergunta de teste.",
            system_prompt="Instruções de teste.",
        )

    trace = get_only_trace(metrics)

    assert trace["status"] == "failure"
    assert trace["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (429, LLMRateLimitError),
        (408, LLMTimeoutError),
        (500, LLMTransientProviderError),
        (502, LLMTransientProviderError),
        (503, LLMTransientProviderError),
        (504, LLMTransientProviderError),
    ],
    ids=[
        "rate-limit",
        "timeout",
        "internal-server-error",
        "bad-gateway",
        "service-unavailable",
        "gateway-timeout",
    ],
)
def test_anthropic_classifies_http_errors_and_records_original_type(
    monkeypatch,
    metrics,
    status_code,
    expected_error,
):
    """Classifica erros HTTP e registra a classe original da exceção."""

    messages = FakeMessages(
        error=FakeAPIStatusError(status_code),
    )

    client = build_client(monkeypatch, messages)

    with pytest.raises(expected_error):
        client.generate_response(
            user_prompt="Pergunta de teste.",
            system_prompt="Instruções de teste.",
        )

    trace = get_only_trace(metrics)

    assert trace["provider"] == "anthropic"
    assert trace["status"] == "failure"
    assert trace["error_type"] == "FakeAPIStatusError"


def test_anthropic_preserves_existing_llm_errors(monkeypatch, metrics):
    """Não converte novamente erros que já pertencem à hierarquia LLMError."""

    original_error = LLMResponseError("Erro interno já classificado.")

    messages = FakeMessages(error=original_error)
    client = build_client(monkeypatch, messages)

    with pytest.raises(LLMResponseError) as exc_info:
        client.generate_response(
            user_prompt="Pergunta de teste.",
            system_prompt="Instruções de teste.",
        )

    assert exc_info.value is original_error

    trace = get_only_trace(metrics)

    assert trace["status"] == "failure"
    assert trace["error_type"] == "LLMResponseError"


def test_anthropic_does_not_store_prompts_or_responses_in_traces(
    monkeypatch,
    metrics,
):
    """Garante que conteúdo sensível não seja armazenado nos traces."""

    prompt_marker = "PRIVATE_PROMPT_MARKER"
    system_marker = "PRIVATE_SYSTEM_MARKER"
    response_marker = "PRIVATE_RESPONSE_MARKER"

    messages = FakeMessages(
        response=build_response(text=response_marker),
    )

    client = build_client(monkeypatch, messages)

    client.generate_response(
        user_prompt=prompt_marker,
        system_prompt=system_marker,
    )

    traces = metrics.get_recent_traces(
        provider="anthropic",
        limit=10,
    )

    serialized_traces = repr(traces)

    assert prompt_marker not in serialized_traces
    assert system_marker not in serialized_traces
    assert response_marker not in serialized_traces
