import pytest

from ai_analyst.llm.factory import create_llm_provider
from ai_analyst.llm.resilience import RetryLLMProvider


class FakeGeminiClient:
    provider_name = "gemini"
    model = "fake-gemini-model"


class FakeOpenAIClient:
    provider_name = "openai"
    model = "fake-openai-model"


class FakeAnthropicClient:
    provider_name = "anthropic"
    model = "fake-anthropic-model"


@pytest.fixture
def fake_providers(monkeypatch):
    monkeypatch.setattr(
        "ai_analyst.llm.factory.GeminiClient",
        FakeGeminiClient,
    )
    monkeypatch.setattr(
        "ai_analyst.llm.factory.OpenAIClient",
        FakeOpenAIClient,
    )
    monkeypatch.setattr(
        "ai_analyst.llm.factory.AnthropicClient",
        FakeAnthropicClient,
    )


def test_factory_creates_gemini_provider(fake_providers):
    provider = create_llm_provider("gemini")

    assert isinstance(provider, RetryLLMProvider)
    assert isinstance(provider.provider, FakeGeminiClient)
    assert provider.provider_name == "gemini"


def test_factory_creates_openai_provider(fake_providers):
    provider = create_llm_provider("openai")

    assert isinstance(provider, RetryLLMProvider)
    assert isinstance(provider.provider, FakeOpenAIClient)
    assert provider.provider_name == "openai"


def test_factory_creates_anthropic_provider(fake_providers):
    provider = create_llm_provider("anthropic")

    assert isinstance(provider, RetryLLMProvider)
    assert isinstance(provider.provider, FakeAnthropicClient)
    assert provider.provider_name == "anthropic"


def test_factory_uses_environment_variable(
    monkeypatch,
    fake_providers,
):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    provider = create_llm_provider()

    assert isinstance(provider.provider, FakeOpenAIClient)


def test_factory_defaults_to_gemini(
    monkeypatch,
    fake_providers,
):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    provider = create_llm_provider()

    assert isinstance(provider.provider, FakeGeminiClient)


@pytest.mark.parametrize(
    ("provider_name", "expected_class"),
    [
        ("  GEMINI  ", FakeGeminiClient),
        ("  OPENAI  ", FakeOpenAIClient),
        ("  ANTHROPIC  ", FakeAnthropicClient),
    ],
)
def test_factory_normalizes_provider_name(
    provider_name,
    expected_class,
    fake_providers,
):
    provider = create_llm_provider(provider_name)

    assert isinstance(provider.provider, expected_class)


@pytest.mark.parametrize(
    "provider_name",
    ["unsupported", "claude", "gpt", "llama"],
)
def test_factory_rejects_unsupported_provider(provider_name):
    with pytest.raises(
        ValueError,
        match="Provedor de IA não suportado",
    ):
        create_llm_provider(provider_name)


def test_factory_rejects_empty_environment_as_default(
    monkeypatch,
    fake_providers,
):
    monkeypatch.setenv("LLM_PROVIDER", "")

    provider = create_llm_provider()

    assert isinstance(provider.provider, FakeGeminiClient)


def test_factory_rejects_whitespace_environment_as_default(
    monkeypatch,
    fake_providers,
):
    monkeypatch.setenv("LLM_PROVIDER", "   ")

    provider = create_llm_provider()

    assert isinstance(provider.provider, FakeGeminiClient)
