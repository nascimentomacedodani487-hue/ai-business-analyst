from types import SimpleNamespace

import pytest

import ai_analyst.llm.openai as openai_module
from ai_analyst.llm.base import (
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTransientProviderError,
)
from ai_analyst.llm.metrics import LLMMetrics


@pytest.fixture
def metrics(monkeypatch):
    instance = LLMMetrics()

    monkeypatch.setattr(
        openai_module,
        "llm_metrics",
        instance,
    )

    yield instance

    instance.reset()


class FakeAPIStatusError(Exception):
    """Simula uma exceção HTTP retornada pelo SDK."""

    def __init__(self, status_code):
        super().__init__(f"Erro HTTP {status_code}")
        self.status_code = status_code


def create_fake_response(
    content="Resposta simulada.",
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
):
    """Cria uma resposta compatível com a estrutura do SDK OpenAI."""

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


def build_openai_client(monkeypatch, create_method):
    """Substitui o SDK por um cliente simulado."""

    fake_sdk_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=create_method,
            )
        )
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-api-key",
    )

    monkeypatch.setattr(
        openai_module,
        "OpenAI",
        lambda api_key: fake_sdk_client,
    )

    return openai_module.OpenAIClient()


# ==========================================================
# INICIALIZAÇÃO
# ==========================================================


def test_openai_client_initialization_without_api_key(monkeypatch):
    """Rejeita inicialização quando a chave não está disponível."""

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="OPENAI_API_KEY",
    ):
        openai_module.OpenAIClient()


def test_openai_client_uses_default_model(monkeypatch):
    """Verifica o modelo padrão do adaptador."""

    client = build_openai_client(
        monkeypatch,
        lambda **kwargs: create_fake_response(),
    )

    assert client.provider_name == "openai"
    assert client.model == "gpt-4.1-mini"


def test_openai_client_accepts_custom_model(monkeypatch):
    """Permite selecionar outro modelo."""

    client = build_openai_client(
        monkeypatch,
        lambda **kwargs: create_fake_response(),
    )

    custom_client = openai_module.OpenAIClient.__new__(openai_module.OpenAIClient)

    custom_client.__dict__.update(client.__dict__)
    custom_client.model = "custom-openai-model"

    assert custom_client.model == "custom-openai-model"


# ==========================================================
# GERAÇÃO DE RESPOSTA
# ==========================================================


def test_openai_success_records_tokens_and_trace(
    monkeypatch,
    metrics,
):
    """Registra consumo, duração e trace de uma chamada bem-sucedida."""

    response = create_fake_response()
    received_arguments = {}

    def fake_create(**kwargs):
        received_arguments.update(kwargs)
        return response

    client = build_openai_client(
        monkeypatch,
        fake_create,
    )

    with metrics.track_analysis(
        "analysis-openai",
        request_id="request-openai",
    ):
        result = client.generate_response(
            user_prompt="Pergunta simulada.",
            system_prompt="Instruções simuladas.",
            max_output_tokens=256,
        )

    assert result == "Resposta simulada."

    assert received_arguments["model"] == "gpt-4.1-mini"
    assert received_arguments["max_completion_tokens"] == 256

    assert received_arguments["messages"] == [
        {
            "role": "system",
            "content": "Instruções simuladas.",
        },
        {
            "role": "user",
            "content": "Pergunta simulada.",
        },
    ]

    snapshot = metrics.snapshot("openai")

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["successful"] == 1
    assert snapshot["requests"]["failed"] == 0

    assert snapshot["tokens"]["input"] == 100
    assert snapshot["tokens"]["output"] == 50
    assert snapshot["tokens"]["total"] == 150

    traces = metrics.get_recent_traces(
        analysis_id="analysis-openai",
        request_id="request-openai",
    )

    assert len(traces) == 1

    event = traces[0]

    assert event["event_type"] == "llm_call"
    assert event["status"] == "success"
    assert event["provider"] == "openai"
    assert event["model"] == "gpt-4.1-mini"
    assert event["analysis_id"] == "analysis-openai"
    assert event["request_id"] == "request-openai"

    assert event["tokens"] == {
        "input": 100,
        "output": 50,
        "thinking": None,
        "total": 150,
    }

    assert event["duration_ms"] >= 0


def test_openai_empty_response_records_failure(
    monkeypatch,
    metrics,
):
    """Registra falha quando a resposta da OpenAI está vazia."""

    client = build_openai_client(
        monkeypatch,
        lambda **kwargs: create_fake_response(content="   "),
    )

    with metrics.track_analysis("analysis-empty-response"):
        with pytest.raises(LLMResponseError):
            client.generate_response(
                user_prompt="Pergunta.",
                system_prompt="Instruções.",
            )

    snapshot = metrics.snapshot("openai")

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["failed"] == 1

    traces = metrics.get_recent_traces(
        analysis_id="analysis-empty-response",
    )

    assert len(traces) == 1

    assert traces[0]["status"] == "failure"
    assert traces[0]["error_type"] == "LLMResponseError"


def test_openai_records_original_exception_type(
    monkeypatch,
    metrics,
):
    """Preserva a classe original da exceção no trace."""

    def fake_create(**kwargs):
        raise RuntimeError("Falha simulada.")

    client = build_openai_client(
        monkeypatch,
        fake_create,
    )

    with metrics.track_analysis("analysis-runtime-error"):
        with pytest.raises(LLMProviderError):
            client.generate_response(
                user_prompt="Pergunta.",
                system_prompt="Instruções.",
            )

    traces = metrics.get_recent_traces(
        analysis_id="analysis-runtime-error",
    )

    assert len(traces) == 1

    event = traces[0]

    assert event["status"] == "failure"
    assert event["error_type"] == "RuntimeError"
    assert event["provider"] == "openai"


# ==========================================================
# CLASSIFICAÇÃO DOS ERROS HTTP
# ==========================================================


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        pytest.param(
            429,
            LLMRateLimitError,
            id="rate-limit",
        ),
        pytest.param(
            408,
            LLMTimeoutError,
            id="timeout",
        ),
        pytest.param(
            500,
            LLMTransientProviderError,
            id="internal-server-error",
        ),
        pytest.param(
            502,
            LLMTransientProviderError,
            id="bad-gateway",
        ),
        pytest.param(
            503,
            LLMTransientProviderError,
            id="service-unavailable",
        ),
        pytest.param(
            504,
            LLMTransientProviderError,
            id="gateway-timeout",
        ),
    ],
)
def test_openai_classifies_http_errors_and_records_original_type(
    monkeypatch,
    metrics,
    status_code,
    expected_exception,
):
    """Verifica classificação e tracing das falhas HTTP."""

    original_error = FakeAPIStatusError(status_code)

    def fake_create(**kwargs):
        raise original_error

    client = build_openai_client(
        monkeypatch,
        fake_create,
    )

    analysis_id = f"analysis-http-{status_code}"

    with metrics.track_analysis(analysis_id):
        with pytest.raises(expected_exception) as captured:
            client.generate_response(
                user_prompt="Pergunta.",
                system_prompt="Instruções.",
            )

    assert captured.value.__cause__ is original_error

    snapshot = metrics.snapshot("openai")

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["failed"] == 1

    traces = metrics.get_recent_traces(
        analysis_id=analysis_id,
        status="failure",
    )

    assert len(traces) == 1

    event = traces[0]

    assert event["error_type"] == "FakeAPIStatusError"
    assert event["provider"] == "openai"
    assert event["status"] == "failure"


def test_openai_preserves_existing_llm_errors(
    monkeypatch,
    metrics,
):
    """Não converte novamente erros já padronizados pelo projeto."""

    original_error = LLMResponseError("Erro de resposta simulado.")

    def fake_create(**kwargs):
        raise original_error

    client = build_openai_client(
        monkeypatch,
        fake_create,
    )

    with metrics.track_analysis("analysis-existing-llm-error"):
        with pytest.raises(LLMResponseError) as captured:
            client.generate_response(
                user_prompt="Pergunta.",
                system_prompt="Instruções.",
            )

    assert captured.value is original_error

    traces = metrics.get_recent_traces(
        analysis_id="analysis-existing-llm-error",
    )

    assert len(traces) == 1
    assert traces[0]["error_type"] == "LLMResponseError"


# ==========================================================
# INTEGRIDADE DAS MÉTRICAS
# ==========================================================


def test_openai_does_not_store_prompts_or_responses_in_traces(
    monkeypatch,
    metrics,
):
    """Garante que conteúdos sensíveis não sejam armazenados."""

    secret_prompt = "CONTEUDO_PRIVADO_PROMPT"
    secret_system = "CONTEUDO_PRIVADO_SYSTEM"
    secret_response = "CONTEUDO_PRIVADO_RESPONSE"

    client = build_openai_client(
        monkeypatch,
        lambda **kwargs: create_fake_response(
            content=secret_response,
        ),
    )

    client.generate_response(
        user_prompt=secret_prompt,
        system_prompt=secret_system,
    )

    event = metrics.get_recent_traces()[0]
    event_text = str(event)

    assert secret_prompt not in event_text
    assert secret_system not in event_text
    assert secret_response not in event_text
