
from ai_analyst.agent.business_analyst import BusinessAnalyst
from ai_analyst.agent.cache import AnalysisCache
import duckdb
import pytest

from ai_analyst.security.guardrails import (
    SQLGuardrailError,
    SQLSyntaxError,
)


def test_business_analyst_flow(monkeypatch):
    """Testa o fluxo completo sem consumir a API real do Gemini."""

    class FakeGeminiClient:
        def generate_response(
            self,
            user_prompt,
            system_prompt,
            max_output_tokens=512,
        ):
            if "Transforme a pergunta" in user_prompt:
                return "SELECT COUNT(*) AS total FROM orders"

            return "Foram encontrados 100 pedidos."

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        FakeGeminiClient,
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: "v1",
    )

    analyst = BusinessAnalyst()

    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        lambda sql: [{"total": 100}],
    )

    response = analyst.analyze("Quantos pedidos tivemos?")

    assert response["question"] == "Quantos pedidos tivemos?"
    assert "SELECT" in response["sql"].upper()
    assert len(response["results"]) > 0
    assert response["total_results"] > 0
    assert isinstance(response["answer"], str)
    assert len(response["answer"]) > 0
    assert response["cache_hit"] is False


def test_business_analyst_initializes_gemini_on_demand(monkeypatch):
    """Verifica se o Gemini só é inicializado quando necessário."""

    class FakeGeminiClient:
        pass

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        FakeGeminiClient,
    )

    analyst = BusinessAnalyst()

    assert analyst.gemini is None
    assert analyst.sql_generator is None
    assert analyst.response_generator is None

    analyst._initialize_ai_components()

    assert isinstance(analyst.gemini, FakeGeminiClient)
    assert analyst.sql_generator is not None
    assert analyst.response_generator is not None


def test_business_analyst_shares_gemini_client(monkeypatch):
    """Verifica se os componentes compartilham o mesmo cliente Gemini."""

    class FakeGeminiClient:
        pass

    fake_gemini = FakeGeminiClient()

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        lambda: fake_gemini,
    )

    analyst = BusinessAnalyst()
    analyst._initialize_ai_components()

    assert analyst.sql_generator.gemini is fake_gemini
    assert analyst.response_generator.gemini is fake_gemini

    assert (
        analyst.sql_generator.gemini
        is analyst.response_generator.gemini
    )


def test_business_analyst_uses_cache_without_initializing_gemini(
    monkeypatch,
):
    """Verifica se o cache evita até mesmo inicializar o Gemini."""

    def unexpected_gemini_initialization():
        raise AssertionError(
            "O Gemini não deveria ser inicializado em um cache hit."
        )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        unexpected_gemini_initialization,
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: "v1",
    )

    cache = AnalysisCache()
    question = "Quantos pedidos tivemos?"

    cached_analysis = {
        "question": question,
        "sql": "SELECT COUNT(*) AS total FROM orders",
        "results": [{"total": 100}],
        "total_results": 1,
        "answer": "Foram encontrados 100 pedidos.",
        "chart": None,
        "cache_hit": False,
    }

    cache.set(
        question=question,
        data_version="v1",
        result=cached_analysis,
    )

    analyst = BusinessAnalyst(cache=cache)
    result = analyst.analyze(question)

    assert result["cache_hit"] is True
    assert result["answer"] == "Foram encontrados 100 pedidos."

    assert analyst.gemini is None
    assert analyst.sql_generator is None
    assert analyst.response_generator is None


def test_business_analyst_invalidates_cache_when_data_changes(
    monkeypatch,
):
    """Verifica se uma nova versão dos dados gera outra análise."""

    class FakeGeminiClient:
        def generate_response(
            self,
            user_prompt,
            system_prompt,
            max_output_tokens=512,
        ):
            if "Transforme a pergunta" in user_prompt:
                return "SELECT COUNT(*) AS total FROM orders"

            return "Análise concluída."

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        FakeGeminiClient,
    )

    current_version = {"value": "v1"}

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: current_version["value"],
    )

    cache = AnalysisCache()
    analyst = BusinessAnalyst(cache=cache)

    calls = {"sql": 0}

    from ai_analyst.agent.sql_generator import SQLGenerator

    original_generate = SQLGenerator.generate

    def counted_generate(self, question):
        calls["sql"] += 1
        return original_generate(self, question)

    monkeypatch.setattr(
        SQLGenerator,
        "generate",
        counted_generate,
    )

    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        lambda sql: [{"total": 100}],
    )

    first_result = analyst.analyze("Quantos pedidos tivemos?")

    assert first_result["cache_hit"] is False

    current_version["value"] = "v2"

    second_result = analyst.analyze("Quantos pedidos tivemos?")

    assert second_result["cache_hit"] is False
    assert calls["sql"] == 2


def _build_analyst_for_sql_recovery(monkeypatch):
    """Cria um analista isolado sem chamadas reais a provedores."""

    class FakeProvider:
        def generate_response(
            self,
            user_prompt,
            system_prompt,
            max_output_tokens=512,
        ):
            return "SELECT 1 AS total"

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        lambda: FakeProvider(),
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: "sql-recovery-test",
    )

    analyst = BusinessAnalyst(cache=AnalysisCache())
    analyst._initialize_ai_components()

    monkeypatch.setattr(
        analyst.response_generator,
        "generate",
        lambda **kwargs: "Analise concluida.",
    )

    return analyst


def test_business_analyst_recovers_initial_sql_syntax_error(monkeypatch):
    analyst = _build_analyst_for_sql_recovery(monkeypatch)

    invalid_sql = "SELECT ("
    valid_sql = "SELECT 1 AS total"
    corrections = []
    executions = []

    def generate(question):
        raise SQLSyntaxError(
            invalid_sql,
            "Sintaxe SQL invalida.",
        )

    def correct_sql(question, failed_sql, error_message):
        corrections.append((failed_sql, error_message))
        return valid_sql

    def execute(sql):
        executions.append(sql)
        return [{"total": 1}]

    monkeypatch.setattr(analyst.sql_generator, "generate", generate)
    monkeypatch.setattr(analyst.sql_generator, "correct_sql", correct_sql)
    monkeypatch.setattr(analyst.sql_executor, "execute", execute)

    result = analyst.analyze("Quantos pedidos tivemos?")

    assert result["sql"] == valid_sql
    assert result["results"] == [{"total": 1}]
    assert len(corrections) == 1
    assert corrections[0][0] == invalid_sql
    assert executions == [valid_sql]


def test_business_analyst_recovers_invalid_correction(monkeypatch):
    analyst = _build_analyst_for_sql_recovery(monkeypatch)

    first_invalid_sql = "SELECT ("
    second_invalid_sql = "SELECT ((1)"
    valid_sql = "SELECT 1 AS total"
    corrections = []

    def generate(question):
        raise SQLSyntaxError(
            first_invalid_sql,
            "Erro de sintaxe inicial.",
        )

    def correct_sql(question, failed_sql, error_message):
        corrections.append(failed_sql)

        if len(corrections) == 1:
            raise SQLSyntaxError(
                second_invalid_sql,
                "Erro de sintaxe na primeira correcao.",
            )

        return valid_sql

    monkeypatch.setattr(analyst.sql_generator, "generate", generate)
    monkeypatch.setattr(analyst.sql_generator, "correct_sql", correct_sql)
    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        lambda sql: [{"total": 1}],
    )

    result = analyst.analyze("Quantos pedidos tivemos?")

    assert result["sql"] == valid_sql
    assert len(corrections) == 2
    assert corrections == [
        first_invalid_sql,
        second_invalid_sql,
    ]


def test_business_analyst_stops_after_max_sql_corrections(monkeypatch):
    analyst = _build_analyst_for_sql_recovery(monkeypatch)

    corrections = []
    executions = []

    def generate(question):
        raise SQLSyntaxError(
            "SELECT (",
            "Erro de sintaxe inicial.",
        )

    def correct_sql(question, failed_sql, error_message):
        corrections.append(failed_sql)
        raise SQLSyntaxError(
            "SELECT ((",
            "Erro de sintaxe persistente.",
        )

    def execute(sql):
        executions.append(sql)
        return []

    monkeypatch.setattr(analyst.sql_generator, "generate", generate)
    monkeypatch.setattr(analyst.sql_generator, "correct_sql", correct_sql)
    monkeypatch.setattr(analyst.sql_executor, "execute", execute)

    with pytest.raises(SQLSyntaxError):
        analyst.analyze("Quantos pedidos tivemos?")

    assert len(corrections) == 2
    assert executions == []


def test_business_analyst_does_not_correct_security_violation(monkeypatch):
    analyst = _build_analyst_for_sql_recovery(monkeypatch)

    corrections = []
    executions = []

    def generate(question):
        raise SQLGuardrailError(
            "Funcao bloqueada: read_csv"
        )

    def correct_sql(question, failed_sql, error_message):
        corrections.append(failed_sql)
        return "SELECT 1 AS total"

    def execute(sql):
        executions.append(sql)
        return [{"total": 1}]

    monkeypatch.setattr(analyst.sql_generator, "generate", generate)
    monkeypatch.setattr(analyst.sql_generator, "correct_sql", correct_sql)
    monkeypatch.setattr(analyst.sql_executor, "execute", execute)

    with pytest.raises(SQLGuardrailError) as exc:
        analyst.analyze("Quantos pedidos tivemos?")

    assert type(exc.value) is SQLGuardrailError
    assert corrections == []
    assert executions == []


def test_business_analyst_recovers_duckdb_execution_error(monkeypatch):
    analyst = _build_analyst_for_sql_recovery(monkeypatch)

    valid_sql = "SELECT 1 AS total"
    corrections = []
    executions = []

    def generate(question):
        return valid_sql

    def correct_sql(question, failed_sql, error_message):
        corrections.append((failed_sql, error_message))
        return valid_sql

    def execute(sql):
        executions.append(sql)

        if len(executions) == 1:
            raise duckdb.ProgrammingError(
                "Binder Error: column not found"
            )

        return [{"total": 1}]

    monkeypatch.setattr(analyst.sql_generator, "generate", generate)
    monkeypatch.setattr(analyst.sql_generator, "correct_sql", correct_sql)
    monkeypatch.setattr(analyst.sql_executor, "execute", execute)

    result = analyst.analyze("Quantos pedidos tivemos?")

    assert result["sql"] == valid_sql
    assert result["results"] == [{"total": 1}]
    assert len(corrections) == 1
    assert len(executions) == 2
    assert corrections[0][0] == valid_sql

