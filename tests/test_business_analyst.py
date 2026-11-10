import duckdb
import pytest

from ai_analyst.agent.business_analyst import BusinessAnalyst
from ai_analyst.agent.cache import AnalysisCache
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

    assert analyst.sql_generator.gemini is analyst.response_generator.gemini


def test_business_analyst_uses_cache_without_initializing_gemini(
    monkeypatch,
):
    """Verifica se o cache evita até mesmo inicializar o Gemini."""

    def unexpected_gemini_initialization():
        raise AssertionError("O Gemini não deveria ser inicializado em um cache hit.")

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
        raise SQLGuardrailError("Funcao bloqueada: read_csv")

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
            raise duckdb.ProgrammingError("Binder Error: column not found")

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


def _build_analyst_for_multi_query(monkeypatch):
    """Cria um analista isolado para testar múltiplas consultas."""

    class FakeProvider:
        def generate_response(
            self,
            user_prompt,
            system_prompt,
            max_output_tokens=512,
        ):
            return "Resposta simulada."

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        lambda: FakeProvider(),
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: "multi-query-test",
    )

    analyst = BusinessAnalyst(cache=AnalysisCache())
    analyst._initialize_ai_components()

    return analyst


def test_business_analyst_executes_multiple_independent_queries(
    monkeypatch,
):
    """Verifica se cada objetivo gera e executa seu próprio SQL."""

    analyst = _build_analyst_for_multi_query(monkeypatch)

    objectives = [
        {
            "id": 1,
            "objective": "Analisar quantidade de pedidos",
            "question": "Quantos pedidos foram realizados?",
        },
        {
            "id": 2,
            "objective": "Analisar faturamento",
            "question": "Qual foi o faturamento total?",
        },
    ]

    monkeypatch.setattr(
        analyst,
        "_requires_query_planning",
        lambda question: True,
    )

    monkeypatch.setattr(
        analyst.query_planner,
        "plan",
        lambda question: objectives,
    )

    generated_sql = [
        "SELECT COUNT(*) AS total_pedidos FROM orders",
        "SELECT SUM(total) AS faturamento FROM orders",
    ]

    generation_calls = []

    def generate(question):
        generation_calls.append(question)
        return generated_sql[len(generation_calls) - 1]

    monkeypatch.setattr(
        analyst.sql_generator,
        "generate",
        generate,
    )

    executed_sql = []

    def execute(sql):
        executed_sql.append(sql)

        if "COUNT" in sql.upper():
            return [{"total_pedidos": 100}]

        return [{"faturamento": 15000.0}]

    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        execute,
    )

    monkeypatch.setattr(
        analyst.response_generator,
        "generate_multi",
        lambda question, queries: "Análise consolidada.",
    )

    result = analyst.analyze("Analise a quantidade de pedidos e o faturamento.")

    assert len(result["queries"]) == 2
    assert result["query_count"] == 2
    assert result["successful_queries"] == 2
    assert result["failed_queries"] == 0
    assert result["partial_success"] is False

    assert len(generation_calls) == 2
    assert len(executed_sql) == 2

    assert executed_sql[0] != executed_sql[1]

    assert result["queries"][0]["status"] == "success"
    assert result["queries"][1]["status"] == "success"

    assert result["queries"][0]["results"] == [{"total_pedidos": 100}]

    assert result["queries"][1]["results"] == [{"faturamento": 15000.0}]

    assert result["answer"] == "Análise consolidada."


def test_business_analyst_preserves_single_query_path(
    monkeypatch,
):
    """Perguntas simples não devem acionar o planejador."""

    analyst = _build_analyst_for_multi_query(monkeypatch)

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: "single-query-test",
    )

    monkeypatch.setattr(
        analyst,
        "_requires_query_planning",
        lambda question: False,
    )

    def unexpected_planning(question):
        raise AssertionError(
            "O planejador não deveria ser chamado para uma pergunta simples."
        )

    monkeypatch.setattr(
        analyst.query_planner,
        "plan",
        unexpected_planning,
    )

    monkeypatch.setattr(
        analyst.sql_generator,
        "generate",
        lambda question: "SELECT COUNT(*) AS total FROM orders",
    )

    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        lambda sql: [{"total": 100}],
    )

    monkeypatch.setattr(
        analyst.response_generator,
        "generate",
        lambda **kwargs: "Foram encontrados 100 pedidos.",
    )

    result = analyst.analyze("Quantos pedidos tivemos?")

    assert result["sql"] == "SELECT COUNT(*) AS total FROM orders"
    assert result["results"] == [{"total": 100}]
    assert result["answer"] == "Foram encontrados 100 pedidos."
    assert result["cache_hit"] is False


def test_business_analyst_multi_query_returns_separate_sql_statements(
    monkeypatch,
):
    """Verifica que os SQLs permanecem separados na resposta."""

    analyst = _build_analyst_for_multi_query(monkeypatch)

    objectives = [
        {
            "id": 1,
            "objective": "Quantidade de pedidos",
            "question": "Quantos pedidos existem?",
        },
        {
            "id": 2,
            "objective": "Quantidade de restaurantes",
            "question": "Quantos restaurantes existem?",
        },
    ]

    monkeypatch.setattr(
        analyst,
        "_requires_query_planning",
        lambda question: True,
    )

    monkeypatch.setattr(
        analyst.query_planner,
        "plan",
        lambda question: objectives,
    )

    sql_statements = [
        "SELECT COUNT(*) AS pedidos FROM orders",
        "SELECT COUNT(*) AS restaurantes FROM restaurants",
    ]

    counter = {"value": 0}

    def generate(question):
        sql = sql_statements[counter["value"]]
        counter["value"] += 1
        return sql

    monkeypatch.setattr(
        analyst.sql_generator,
        "generate",
        generate,
    )

    executed = []

    def execute(sql):
        executed.append(sql)
        return [{"total": 10}]

    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        execute,
    )

    monkeypatch.setattr(
        analyst.response_generator,
        "generate_multi",
        lambda question, queries: "Análise concluída.",
    )

    result = analyst.analyze("Analise os pedidos e os restaurantes.")

    assert len(result["queries"]) == 2

    for query in result["queries"]:
        assert query["sql"].strip().upper().startswith("SELECT")

    assert len(executed) == 2

    # Não pode existir um SQL combinado com múltiplas instruções.
    assert all(";" not in sql for sql in executed)

    # O campo legado contém apenas o SQL da consulta principal.
    assert result["sql"] == result["queries"][0]["sql"]


def test_business_analyst_continues_after_partial_query_failure(
    monkeypatch,
):
    """Uma consulta com falha não deve interromper as demais."""

    analyst = _build_analyst_for_multi_query(monkeypatch)

    objectives = [
        {
            "id": 1,
            "objective": "Analisar quantidade de pedidos",
            "question": "Quantos pedidos foram realizados?",
        },
        {
            "id": 2,
            "objective": "Analisar faturamento",
            "question": "Qual foi o faturamento total?",
        },
        {
            "id": 3,
            "objective": "Analisar restaurantes",
            "question": "Quantos restaurantes existem?",
        },
    ]

    monkeypatch.setattr(
        analyst,
        "_requires_query_planning",
        lambda question: True,
    )

    monkeypatch.setattr(
        analyst.query_planner,
        "plan",
        lambda question: objectives,
    )

    execution_calls = []

    def fake_execute_planned_query(
        planned_query,
        record_stage,
        run_stage,
    ):
        query_id = planned_query["id"]
        objective = planned_query
        question = planned_query["question"]

        execution_calls.append(query_id)

        if query_id == 2:
            return {
                "id": query_id,
                "objective": objective["objective"],
                "question": question,
                "sql": "SELECT SUM(total) FROM orders",
                "results": [],
                "total_results": 0,
                "status": "failure",
                "error": "Erro simulado de execução SQL.",
                "chart": None,
                "metrics": {
                    "sql_generation_ms": 10,
                    "sql_execution_ms": 5,
                    "total_time_ms": 15,
                },
            }

        return {
            "id": query_id,
            "objective": objective["objective"],
            "question": question,
            "sql": (
                "SELECT COUNT(*) FROM orders"
                if query_id == 1
                else "SELECT COUNT(*) FROM restaurants"
            ),
            "results": [{"total": 100}],
            "total_results": 1,
            "status": "success",
            "error": None,
            "chart": None,
            "metrics": {
                "sql_generation_ms": 10,
                "sql_execution_ms": 5,
                "total_time_ms": 15,
            },
        }

    monkeypatch.setattr(
        analyst,
        "_execute_planned_query",
        fake_execute_planned_query,
    )

    received_queries = []

    def fake_generate_multi(question, queries):
        received_queries.extend(queries)
        return "Análise concluída com uma falha parcial."

    monkeypatch.setattr(
        analyst.response_generator,
        "generate_multi",
        fake_generate_multi,
    )

    result = analyst.analyze("Analise os pedidos, faturamento e restaurantes.")

    assert execution_calls == [1, 2, 3]

    assert len(result["queries"]) == 3
    assert result["query_count"] == 3

    assert result["successful_queries"] == 2
    assert result["failed_queries"] == 1
    assert result["partial_success"] is True

    assert result["queries"][0]["status"] == "success"
    assert result["queries"][1]["status"] == "failure"
    assert result["queries"][2]["status"] == "success"

    assert result["queries"][0]["results"] == [{"total": 100}]
    assert result["queries"][1]["results"] == []
    assert result["queries"][2]["results"] == [{"total": 100}]

    assert len(received_queries) == 3

    assert result["answer"] == ("Análise concluída com uma falha parcial.")


def test_unsupported_product_query_does_not_replace_requested_analysis(
    monkeypatch,
):
    """
    Uma análise de produto indisponível deve ser bloqueada antes
    da geração SQL, sem impedir as outras consultas planejadas.
    """

    analyst = _build_analyst_for_multi_query(monkeypatch)

    objectives = [
        {
            "id": 1,
            "objective": "Analisar faturamento mensal",
            "question": "Qual foi o faturamento por mês?",
        },
        {
            "id": 2,
            "objective": "Analisar pedidos por status",
            "question": "Quantos pedidos existem por status?",
        },
        {
            "id": 3,
            "objective": "Identificar os top 10 produtos com maior faturamento",
            "question": "Quais são os 10 produtos com maior faturamento?",
        },
    ]

    monkeypatch.setattr(
        analyst,
        "_requires_query_planning",
        lambda question: True,
    )

    monkeypatch.setattr(
        analyst.query_planner,
        "plan",
        lambda question: objectives,
    )

    generated_questions = []

    def fake_generate(question):
        generated_questions.append(question)
        return "SELECT COUNT(*) AS total_pedidos FROM orders"

    monkeypatch.setattr(
        analyst.sql_generator,
        "generate",
        fake_generate,
    )

    executed_sql = []

    def fake_execute(sql):
        executed_sql.append(sql)
        return [{"total_pedidos": 100}]

    monkeypatch.setattr(
        analyst.sql_executor,
        "execute",
        fake_execute,
    )

    monkeypatch.setattr(
        analyst.response_generator,
        "generate_multi",
        lambda question, queries: "Resumo executivo das análises.",
    )

    result = analyst.analyze(
        "Analise o faturamento mensal, os pedidos por status "
        "e identifique os 10 produtos com maior faturamento."
    )

    # Somente as duas análises suportadas devem gerar SQL.
    assert len(generated_questions) == 2
    assert len(executed_sql) == 2

    # Contagem geral.
    assert result["query_count"] == 3
    assert result["successful_queries"] == 2
    assert result["failed_queries"] == 0
    assert result["unsupported_queries"] == 1
    assert result["partial_success"] is True

    # As duas consultas viáveis foram concluídas.
    assert result["queries"][0]["status"] == "success"
    assert result["queries"][1]["status"] == "success"

    # A análise de produtos foi bloqueada antes da geração SQL.
    unsupported = result["queries"][2]

    assert unsupported["status"] == "failure"
    assert unsupported["failure_type"] == "unsupported_capability"
    assert unsupported["sql"] is None
    assert unsupported["results"] == []
    assert unsupported["total_results"] == 0
    assert unsupported["chart"] is None
    assert "orders.items_total" in unsupported["error"]
    assert "Nenhuma consulta substituta" in unsupported["error"]

    assert unsupported["metrics"]["sql_generation_ms"] == 0.0
    assert unsupported["metrics"]["sql_execution_ms"] == 0.0

    # A etapa da limitação deve aparecer na observabilidade.
    capability_step = next(
        step
        for step in result["steps"]
        if step["name"] == "query_3_capability_validation"
    )

    assert capability_step["status"] == "unsupported"

    # A síntese final deve preservar a limitação.
    assert "Limitações dos dados:" in result["answer"]
    assert "orders.items_total" in result["answer"]
