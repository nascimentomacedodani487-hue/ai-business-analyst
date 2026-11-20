import json

import pytest

from ai_analyst.agent.sql_generator import SQLGenerator
from ai_analyst.security.guardrails import SQLGuardrailError


class FakeGeminiClient:
    """Simula respostas do Gemini para os testes."""

    def __init__(self, response):
        self.response = response
        self.last_prompt = None
        self.last_system_prompt = None
        self.last_max_output_tokens = None

    def generate_response(
        self,
        user_prompt,
        system_prompt,
        max_output_tokens=512,
    ):
        self.last_prompt = user_prompt
        self.last_system_prompt = system_prompt
        self.last_max_output_tokens = max_output_tokens
        return self.response


class FakeStructuredGeminiClient(FakeGeminiClient):
    """Simula um provedor com suporte a respostas JSON estruturadas."""

    def __init__(self, response):
        super().__init__(response)
        self.last_response_schema = None

    def generate_json_response(
        self,
        user_prompt,
        system_prompt,
        response_schema,
        max_output_tokens=512,
    ):
        self.last_prompt = user_prompt
        self.last_system_prompt = system_prompt
        self.last_response_schema = response_schema
        self.last_max_output_tokens = max_output_tokens
        return self.response


def make_behavior_response(
    behavior_type,
    *,
    sql=None,
    reason="Decisão fundamentada no schema disponível.",
    clarification_question=None,
    limitations=None,
):
    """Cria uma resposta comportamental JSON para os testes."""

    return json.dumps(
        {
            "decision": {
                "type": behavior_type,
                "reason": reason,
                "clarification_question": clarification_question,
                "limitations": limitations or [],
            },
            "sql": sql,
        },
        ensure_ascii=False,
    )


def test_generate_sql():
    """Verifica se a pergunta é transformada em SQL válido."""

    fake_gemini = FakeGeminiClient(
        "SELECT city, COUNT(*) AS total FROM orders GROUP BY city"
    )

    generator = SQLGenerator(gemini=fake_gemini)

    sql = generator.generate("Qual cidade teve mais pedidos entregues?")

    assert isinstance(sql, str)
    assert "SELECT" in sql.upper()
    assert "FROM orders" in sql
    assert "GROUP BY city" in sql

    assert "Qual cidade teve mais pedidos entregues?" in fake_gemini.last_prompt

    assert fake_gemini.last_system_prompt
    assert fake_gemini.last_max_output_tokens == 512


def test_generate_sql_rejects_empty_question():
    """Verifica se perguntas vazias são rejeitadas."""

    fake_gemini = FakeGeminiClient("SELECT 1")
    generator = SQLGenerator(gemini=fake_gemini)

    with pytest.raises(ValueError, match="pergunta não pode estar vazia"):
        generator.generate("")

    assert fake_gemini.last_prompt is None


def test_generate_sql_rejects_unsafe_query():
    """Verifica se consultas perigosas são bloqueadas."""

    fake_gemini = FakeGeminiClient("DELETE FROM orders")

    generator = SQLGenerator(gemini=fake_gemini)

    with pytest.raises(SQLGuardrailError):
        generator.generate("Apague os pedidos.")


def test_correct_sql_successfully():
    """Verifica a correção de uma consulta SQL."""

    fake_gemini = FakeGeminiClient(
        "SELECT city, COUNT(*) AS total FROM orders GROUP BY city"
    )

    generator = SQLGenerator(provider=fake_gemini)

    sql = generator.correct_sql(
        question="Qual cidade teve mais pedidos?",
        failed_sql="SELECT os.total FROM orders",
        error_message='Referenced table "os" not found',
    )

    assert "SELECT" in sql.upper()
    assert "FROM orders" in sql
    assert "Qual cidade teve mais pedidos?" in fake_gemini.last_prompt
    assert "SELECT os.total FROM orders" in fake_gemini.last_prompt
    assert 'Referenced table "os" not found' in fake_gemini.last_prompt
    assert fake_gemini.last_system_prompt
    assert fake_gemini.last_max_output_tokens == 512


@pytest.mark.parametrize(
    ("question", "failed_sql", "error_message"),
    [
        ("", "SELECT 1", "Binder Error"),
        ("Pergunta válida", "", "Binder Error"),
        ("Pergunta válida", "SELECT 1", ""),
    ],
)
def test_correct_sql_rejects_empty_parameters(
    question,
    failed_sql,
    error_message,
):
    """Verifica se os parâmetros obrigatórios são validados."""

    fake_gemini = FakeGeminiClient("SELECT 1")
    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError):
        generator.correct_sql(
            question=question,
            failed_sql=failed_sql,
            error_message=error_message,
        )

    assert fake_gemini.last_prompt is None


def test_correct_sql_rejects_unsafe_query():
    """Verifica se uma correção perigosa é bloqueada."""

    fake_gemini = FakeGeminiClient("SELECT * FROM read_csv('arquivo.csv')")

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(SQLGuardrailError):
        generator.correct_sql(
            question="Quantos pedidos existem?",
            failed_sql="SELECT os.total FROM orders",
            error_message="Referenced table os not found",
        )

    assert fake_gemini.last_prompt is not None


# ==========================================================
# GERAÇÃO COM AVALIAÇÃO COMPORTAMENTAL
# ==========================================================


def test_generate_with_behavior_returns_valid_sql_for_answer():
    """Verifica a decisão answer e a validação do SQL gerado."""

    fake_gemini = FakeGeminiClient(
        make_behavior_response(
            "answer",
            sql="SELECT city, COUNT(*) AS total FROM orders GROUP BY city",
        )
    )

    generator = SQLGenerator(provider=fake_gemini)

    result = generator.generate_with_behavior("Quantos pedidos existem por cidade?")

    assert isinstance(result, dict)
    assert "sql" in result
    assert "behavior" in result

    assert result["sql"].startswith("SELECT")
    assert "FROM orders" in result["sql"]

    assert result["behavior"]["type"] == "answer"
    assert result["behavior"]["reason"]
    assert result["behavior"]["clarification_question"] is None
    assert result["behavior"]["limitations"] == []
    assert result["behavior"]["supported_information"] == []
    assert result["behavior"]["sql_executed"] is False

    assert fake_gemini.last_prompt is not None
    assert "Quantos pedidos existem por cidade?" in fake_gemini.last_prompt
    assert fake_gemini.last_system_prompt
    assert fake_gemini.last_max_output_tokens == 1024


def test_generate_with_behavior_returns_clarification_without_sql():
    """Verifica o comportamento ask_clarification."""

    fake_gemini = FakeGeminiClient(
        make_behavior_response(
            "ask_clarification",
            reason="O período da análise não foi especificado.",
            clarification_question="Qual período você deseja analisar?",
        )
    )

    generator = SQLGenerator(provider=fake_gemini)

    result = generator.generate_with_behavior("Analise o desempenho dos pedidos.")

    assert result["sql"] is None
    assert result["behavior"]["type"] == "ask_clarification"
    assert result["behavior"]["reason"]
    assert (
        result["behavior"]["clarification_question"]
        == "Qual período você deseja analisar?"
    )
    assert result["behavior"]["limitations"] == []
    assert result["behavior"]["sql_executed"] is False


def test_generate_with_behavior_returns_abstention_without_sql():
    """Verifica o comportamento abstain com limitações explícitas."""

    fake_gemini = FakeGeminiClient(
        make_behavior_response(
            "abstain",
            reason="A solicitação depende de informações externas.",
            limitations=[
                "O schema não possui dados meteorológicos.",
            ],
        )
    )

    generator = SQLGenerator(provider=fake_gemini)

    result = generator.generate_with_behavior(
        "Qual era a temperatura durante cada entrega?"
    )

    assert result["sql"] is None
    assert result["behavior"]["type"] == "abstain"
    assert result["behavior"]["reason"]
    assert result["behavior"]["clarification_question"] is None
    assert len(result["behavior"]["limitations"]) == 1
    assert result["behavior"]["sql_executed"] is False


def test_generate_with_behavior_supports_structured_provider():
    """Verifica o caminho generate_json_response do provedor."""

    fake_gemini = FakeStructuredGeminiClient(
        {
            "decision": {
                "type": "answer",
                "reason": "A pergunta pode ser respondida pelo schema.",
                "clarification_question": None,
                "limitations": [],
            },
            "sql": "SELECT COUNT(*) AS total FROM orders",
        }
    )

    generator = SQLGenerator(provider=fake_gemini)

    result = generator.generate_with_behavior("Quantos pedidos existem?")

    assert result["behavior"]["type"] == "answer"
    assert result["sql"] == "SELECT COUNT(*) AS total FROM orders"

    assert fake_gemini.last_response_schema is not None
    assert "decision" in fake_gemini.last_response_schema["properties"]
    assert "sql" in fake_gemini.last_response_schema["properties"]
    assert fake_gemini.last_max_output_tokens == 1024


def test_generate_with_behavior_rejects_empty_question():
    """Verifica se perguntas vazias são rejeitadas antes da chamada."""

    fake_gemini = FakeGeminiClient("")

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError, match="pergunta não pode estar vazia"):
        generator.generate_with_behavior("   ")

    assert fake_gemini.last_prompt is None


def test_generate_with_behavior_rejects_invalid_json():
    """Verifica se respostas JSON inválidas são rejeitadas."""

    fake_gemini = FakeGeminiClient("{json inválido")

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError, match="JSON comportamental inválido"):
        generator.generate_with_behavior("Quantos pedidos existem?")


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "   ",
        [],
        123,
    ],
)
def test_generate_with_behavior_rejects_invalid_response_types(response):
    """Verifica se respostas vazias ou com tipos inválidos são rejeitadas."""

    fake_gemini = FakeGeminiClient(response)

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError):
        generator.generate_with_behavior("Quantos pedidos existem?")


def test_generate_with_behavior_rejects_markdown_fence_content_errors():
    """Verifica que JSON em Markdown é interpretado, mas continua validado."""

    fake_gemini = FakeGeminiClient(
        "```json\n"
        + make_behavior_response(
            "answer",
            sql="DELETE FROM orders",
        )
        + "\n```"
    )

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(SQLGuardrailError):
        generator.generate_with_behavior("Apague os pedidos.")


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        (
            {
                "decision": {
                    "type": "unknown",
                    "reason": "Decisão inválida.",
                    "clarification_question": None,
                    "limitations": [],
                },
                "sql": None,
            },
            "Comportamento não suportado",
        ),
        (
            {
                "decision": {
                    "type": "answer",
                    "reason": "Sem SQL.",
                    "clarification_question": None,
                    "limitations": [],
                },
                "sql": None,
            },
            "precisa apresentar uma consulta SQL",
        ),
        (
            {
                "decision": {
                    "type": "ask_clarification",
                    "reason": "Falta informação.",
                    "clarification_question": ("Qual período deseja analisar?"),
                    "limitations": [],
                },
                "sql": "SELECT 1",
            },
            "não pode gerar SQL",
        ),
        (
            {
                "decision": {
                    "type": "ask_clarification",
                    "reason": "Falta informação.",
                    "clarification_question": None,
                    "limitations": [],
                },
                "sql": None,
            },
            "precisa apresentar uma pergunta",
        ),
        (
            {
                "decision": {
                    "type": "abstain",
                    "reason": "Informação indisponível.",
                    "clarification_question": None,
                    "limitations": [],
                },
                "sql": None,
            },
            "precisa explicar pelo menos",
        ),
        (
            {
                "decision": {
                    "type": "answer",
                    "reason": "A pergunta pode ser respondida.",
                    "clarification_question": ("Qual período deseja analisar?"),
                    "limitations": [],
                },
                "sql": "SELECT 1",
            },
            "não pode solicitar esclarecimentos",
        ),
        (
            {
                "decision": {
                    "type": "answer",
                    "reason": "",
                    "clarification_question": None,
                    "limitations": [],
                },
                "sql": "SELECT 1",
            },
            "precisa apresentar uma justificativa",
        ),
    ],
)
def test_generate_with_behavior_rejects_inconsistent_decisions(
    response,
    expected_error,
):
    """Verifica a coerência entre comportamento, SQL e evidências."""

    fake_gemini = FakeGeminiClient(json.dumps(response))

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError, match=expected_error):
        generator.generate_with_behavior("Analise os pedidos.")


def test_generate_with_behavior_rejects_unsafe_sql():
    """Verifica que o SQL comportamental também passa pelos Guardrails."""

    fake_gemini = FakeGeminiClient(
        make_behavior_response(
            "answer",
            sql="SELECT * FROM read_csv('arquivo.csv')",
        )
    )

    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(SQLGuardrailError):
        generator.generate_with_behavior("Analise os dados do arquivo externo.")


def test_generate_with_behavior_rejects_additional_top_level_fields():
    """Verifica se propriedades não previstas são rejeitadas."""

    response = {
        "decision": {
            "type": "answer",
            "reason": "A pergunta pode ser respondida.",
            "clarification_question": None,
            "limitations": [],
        },
        "sql": "SELECT 1",
        "extra": "campo não permitido",
    }

    fake_gemini = FakeGeminiClient(json.dumps(response))
    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError, match="somente decision e sql"):
        generator.generate_with_behavior("Quantos pedidos existem?")


def test_generate_with_behavior_rejects_missing_decision_fields():
    """Verifica se campos obrigatórios da decisão são exigidos."""

    response = {
        "decision": {
            "type": "answer",
            "reason": "A pergunta pode ser respondida.",
            "clarification_question": None,
        },
        "sql": "SELECT 1",
    }

    fake_gemini = FakeGeminiClient(json.dumps(response))
    generator = SQLGenerator(provider=fake_gemini)

    with pytest.raises(ValueError, match="exatamente"):
        generator.generate_with_behavior("Quantos pedidos existem?")
