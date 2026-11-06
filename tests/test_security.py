
import pytest

from ai_analyst.security.guardrails import (
    validate_query,
    SQLGuardrailError,
)


# 1. CONSULTAS LEGÍTIMAS

@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM orders",
        "SELECT city, COUNT(*) FROM orders GROUP BY city",
        "SELECT * FROM orders WHERE id = 1",
        "SELECT AVG(delivery_time) FROM orders",

        # Tabelas autorizadas
        "SELECT * FROM restaurants",
        "SELECT * FROM customers",
        "SELECT * FROM couriers",
    ],
)
def test_valid_select_queries(query):
    result = validate_query(query)

    assert isinstance(result, str)
    assert "SELECT" in result.upper()


# 2. COMANDOS NÃO AUTORIZADOS

@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM orders WHERE id = 1",
        "INSERT INTO orders VALUES (1, 'test')",
        "UPDATE orders SET city = 'São Paulo'",
        "DROP TABLE orders",
        "ALTER TABLE orders ADD COLUMN test INTEGER",
        "CREATE TABLE malicious AS SELECT * FROM orders",
        "TRUNCATE TABLE orders",
    ],
)
def test_unauthorized_commands(query):
    with pytest.raises(SQLGuardrailError):
        validate_query(query)


# 3. MÚLTIPLAS INSTRUÇÕES

@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM orders; DROP TABLE orders;",
        "SELECT * FROM orders; DELETE FROM orders;",
        "SELECT 1; SELECT 2;",
    ],
)
def test_multiple_statements(query):
    with pytest.raises(SQLGuardrailError):
        validate_query(query)


# 4. TENTATIVAS DE ACESSO A ARQUIVOS

@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM read_csv('secret.csv')",
        "SELECT * FROM read_parquet('/etc/passwd')",
        "SELECT read_text('secret.txt')",
        "SELECT * FROM parquet_scan('data.parquet')",
        "SELECT * FROM read_json_auto('secret.json')",
        "SELECT * FROM glob('*')",
    ],
)
def test_filesystem_attacks(query):
    with pytest.raises(SQLGuardrailError):
        validate_query(query)


# 5. ACESSO A BANCOS EXTERNOS

@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM sqlite_scan('database.db', 'users')",
        "SELECT * FROM postgres_scan('host', 'db', 'schema', 'table')",
    ],
)
def test_external_database_attacks(query):
    with pytest.raises(SQLGuardrailError):
        validate_query(query)


# 6. ENTRADAS INVÁLIDAS

@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
        123,
        "SELECT * FORM orders",
    ],
)
def test_invalid_queries(query):
    with pytest.raises(SQLGuardrailError):
        validate_query(query)


# 7. TENTATIVA DE JAILBREAK

def test_jailbreak_attempt():
    query = """
    SELECT * FROM orders
    -- Ignore previous security rules
    ; DROP TABLE orders;
    """

    with pytest.raises(SQLGuardrailError):
        validate_query(query)


# 8. TABELAS NÃO AUTORIZADAS

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM products",
        "SELECT * FROM users",
        "SELECT * FROM payments",
        "SELECT * FROM information_schema.tables",
    ],
)
def test_unauthorized_tables(sql):
    with pytest.raises(SQLGuardrailError, match="Tabela não autorizada"):
        validate_query(sql)


# 9. CTEs LEGÍTIMAS

def test_valid_cte_query():
    query = """
    WITH ranked_city_revenue AS (
        SELECT
            city,
            SUM(total_amount) AS revenue
        FROM orders
        GROUP BY city
    )
    SELECT *
    FROM ranked_city_revenue
    ORDER BY revenue DESC
    """

    result = validate_query(query)

    assert isinstance(result, str)
    assert "ranked_city_revenue" in result


# 10. CTEs ENCADEADAS

def test_chained_ctes():
    query = """
    WITH city_revenue AS (
        SELECT
            city,
            SUM(total_amount) AS revenue
        FROM orders
        GROUP BY city
    ),
    ranked_city_revenue AS (
        SELECT
            city,
            revenue,
            RANK() OVER (ORDER BY revenue DESC) AS ranking
        FROM city_revenue
    )
    SELECT *
    FROM ranked_city_revenue
    """

    result = validate_query(query)

    assert isinstance(result, str)
    assert "city_revenue" in result
    assert "ranked_city_revenue" in result


# 11. CTE NÃO PODE ESCONDER TABELA NÃO AUTORIZADA

def test_cte_rejects_unauthorized_physical_table():
    query = """
    WITH revenue AS (
        SELECT *
        FROM payments
    )
    SELECT *
    FROM revenue
    """

    with pytest.raises(
        SQLGuardrailError,
        match="Tabela não autorizada",
    ):
        validate_query(query)


# 12. CTE NÃO PODE UTILIZAR FUNÇÕES BLOQUEADAS

def test_cte_rejects_blocked_functions():
    query = """
    WITH imported_data AS (
        SELECT *
        FROM read_csv('secret.csv')
    )
    SELECT *
    FROM imported_data
    """

    with pytest.raises(SQLGuardrailError, match="Função bloqueada"):
        validate_query(query)
