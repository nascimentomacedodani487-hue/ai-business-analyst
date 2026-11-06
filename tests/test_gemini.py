
from types import SimpleNamespace

import pytest

from ai_analyst.llm.base import (
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from ai_analyst.llm.gemini import GeminiClient


class FakeHTTPError(Exception):
    """Simula um erro HTTP retornado pelo provedor."""

    def __init__(self, message, code=None, status_code=None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def create_gemini(monkeypatch, error=None, response_text=None):
    """Cria um GeminiClient com SDK simulado."""

    class FakeModels:
        def generate_content(self, **kwargs):
            if error is not None:
                raise error

            return SimpleNamespace(text=response_text)

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-api-key")

    monkeypatch.setattr(
        "ai_analyst.llm.gemini.genai.Client",
        FakeClient,
    )

    return GeminiClient()


def test_gemini_connection(monkeypatch):
    """Testa a comunicação com o SDK sem chamar a API real."""

    calls = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.update(kwargs)

            return SimpleNamespace(
                text="Sou o AI Business Analyst."
            )

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "fake-api-key")

    monkeypatch.setattr(
        "ai_analyst.llm.gemini.genai.Client",
        FakeClient,
    )

    gemini = GeminiClient()

    response = gemini.generate_response(
        user_prompt="Explique brevemente qual é sua função.",
        system_prompt="Você é um analista de negócios.",
    )

    assert isinstance(response, str)
    assert response == "Sou o AI Business Analyst."

    assert calls["api_key"] == "fake-api-key"
    assert calls["model"] == "gemini-2.5-flash"
    assert calls["contents"] == (
        "Explique brevemente qual é sua função."
    )
    assert calls["config"].system_instruction == (
        "Você é um analista de negócios."
    )
    assert calls["config"].temperature == 0.1


def test_gemini_rejects_missing_api_key(monkeypatch):
    """Verifica se o cliente rejeita uma chave ausente."""

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_gemini_rejects_empty_response(monkeypatch):
    """Verifica o tratamento de respostas vazias do Gemini."""

    gemini = create_gemini(
        monkeypatch,
        response_text=None,
    )

    with pytest.raises(LLMResponseError, match="resposta vazia"):
        gemini.generate_response(
            user_prompt="Teste",
            system_prompt="Instruções",
        )


def test_gemini_rejects_whitespace_response(monkeypatch):
    """Verifica se respostas contendo apenas espaços são rejeitadas."""

    gemini = create_gemini(
        monkeypatch,
        response_text="   ",
    )

    with pytest.raises(LLMResponseError, match="resposta vazia"):
        gemini.generate_response(
            user_prompt="Teste",
            system_prompt="Instruções",
        )


def test_gemini_maps_rate_limit_error(monkeypatch):
    """Converte HTTP 429 em erro padronizado de limite."""

    error = FakeHTTPError(
        "Quota exceeded",
        code=429,
    )

    gemini = create_gemini(monkeypatch, error=error)

    with pytest.raises(LLMRateLimitError) as captured:
        gemini.generate_response("Teste", "Instruções")

    assert captured.value.__cause__ is error


@pytest.mark.parametrize(
    "status_code",
    [500, 502, 503, 504],
)
def test_gemini_maps_server_errors(monkeypatch, status_code):
    """Converte erros temporários do servidor em erro do provedor."""

    error = FakeHTTPError(
        "Server unavailable",
        code=status_code,
    )

    gemini = create_gemini(monkeypatch, error=error)

    with pytest.raises(LLMProviderError) as captured:
        gemini.generate_response("Teste", "Instruções")

    assert not isinstance(captured.value, LLMRateLimitError)
    assert not isinstance(captured.value, LLMTimeoutError)
    assert captured.value.__cause__ is error


def test_gemini_maps_timeout_exception(monkeypatch):
    """Converte TimeoutError em erro padronizado de timeout."""

    error = TimeoutError("Connection timed out")

    gemini = create_gemini(monkeypatch, error=error)

    with pytest.raises(LLMTimeoutError) as captured:
        gemini.generate_response("Teste", "Instruções")

    assert captured.value.__cause__ is error


def test_gemini_maps_http_408_to_timeout(monkeypatch):
    """Converte HTTP 408 em erro padronizado de timeout."""

    error = FakeHTTPError(
        "Request timeout",
        status_code=408,
    )

    gemini = create_gemini(monkeypatch, error=error)

    with pytest.raises(LLMTimeoutError) as captured:
        gemini.generate_response("Teste", "Instruções")

    assert captured.value.__cause__ is error


def test_gemini_maps_unexpected_error(monkeypatch):
    """Converte erros desconhecidos em erro do provedor."""

    error = RuntimeError("Unexpected failure")

    gemini = create_gemini(monkeypatch, error=error)

    with pytest.raises(LLMProviderError) as captured:
        gemini.generate_response("Teste", "Instruções")

    assert captured.value.__cause__ is error


def test_gemini_preserves_standardized_errors(monkeypatch):
    """Não converte novamente erros já padronizados."""

    error = LLMResponseError("Resposta vazia")

    gemini = create_gemini(monkeypatch, error=error)

    with pytest.raises(LLMResponseError) as captured:
        gemini.generate_response("Teste", "Instruções")

    assert captured.value is error
    assert isinstance(captured.value, LLMError)
