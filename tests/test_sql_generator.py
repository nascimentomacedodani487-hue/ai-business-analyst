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

    assert "Qual cidade teve mais pedidos entregues?" in (fake_gemini.last_prompt)

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
