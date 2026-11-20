from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from ai_analyst.security.guardrails import (
    SQLGuardrailError,
    SQLSyntaxError,
    validate_query,
)

# ============================================================
# CONSULTAS LEGÍTIMAS
# ============================================================


@pytest.mark.parametrize(
    "table_name",
    [
        "orders",
        "restaurants",
        "customers",
        "couriers",
    ],
)
def test_accepts_authorized_tables(table_name: str):
    """Aceita consultas às quatro tabelas autorizadas."""

    sql = f"SELECT * FROM {table_name}"

    result = validate_query(sql)

    assert isinstance(result, str)
    assert result.strip()


def test_accepts_read_only_cte():
    """Aceita CTE que contém exclusivamente SELECT."""

    sql = """
        WITH order_counts AS (
            SELECT status, COUNT(*) AS total
            FROM orders
            GROUP BY status
        )
        SELECT status, total
        FROM order_counts
    """

    result = validate_query(sql)

    assert isinstance(result, str)
    assert "order_counts" in result.lower()
    assert "orders" in result.lower()


def test_returns_normalized_sql():
    """Retorna a consulta normalizada pelo SQLGlot."""

    sql = "select count(*) as total from orders"

    result = validate_query(sql)

    assert result == "SELECT COUNT(*) AS total FROM orders"


# ============================================================
# VALIDAÇÃO DE ENTRADA
# ============================================================


@pytest.mark.parametrize(
    "sql",
    [
        "",
        " ",
        "\n\t",
        None,
        123,
    ],
)
def test_rejects_empty_or_invalid_input(sql):
    """Rejeita entradas vazias ou que não sejam strings."""

    with pytest.raises(SQLGuardrailError):
        validate_query(sql)


def test_invalid_sql_raises_syntax_error():
    """Identifica erros de sintaxe SQL."""

    sql = "SELECT ( FROM orders"

    with pytest.raises(SQLSyntaxError):
        validate_query(sql)


# ============================================================
# RESTRIÇÃO DE COMANDOS
# ============================================================


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders (status) VALUES ('pending')",
        "UPDATE orders SET status = 'delivered'",
        "DELETE FROM orders",
        "CREATE TABLE unauthorized_table (id INTEGER)",
    ],
)
def test_rejects_non_select_statements(sql: str):
    """Rejeita comandos cuja raiz não seja SELECT."""

    with pytest.raises(SQLGuardrailError):
        validate_query(sql)


def test_rejects_multiple_statements():
    """Rejeita múltiplas instruções na mesma entrada."""

    sql = """
        SELECT COUNT(*) FROM orders;
        SELECT COUNT(*) FROM customers;
    """

    with pytest.raises(SQLGuardrailError):
        validate_query(sql)


# ============================================================
# RESTRIÇÃO DE TABELAS E RECURSOS EXTERNOS
# ============================================================


def test_rejects_unauthorized_table():
    """Bloqueia tabelas que não fazem parte do catálogo autorizado."""

    sql = "SELECT * FROM confidential_data"

    with pytest.raises(
        SQLGuardrailError,
        match="Tabela não autorizada",
    ):
        validate_query(sql)


def test_rejects_external_schema():
    """Bloqueia referências qualificadas por schema."""

    sql = "SELECT * FROM external_schema.orders"

    with pytest.raises(
        SQLGuardrailError,
        match="Tabela não autorizada",
    ):
        validate_query(sql)


def test_rejects_file_access_functions():
    """Bloqueia funções de leitura de arquivos externos."""

    sql = "SELECT * FROM read_csv('external_data.csv')"

    with pytest.raises(
        SQLGuardrailError,
        match="Função bloqueada",
    ):
        validate_query(sql)


# ============================================================
# SEGURANÇA DE CTEs
# ============================================================


def test_rejects_data_modifying_update_inside_cte():
    """
    Reproduz uma tentativa de escrita dentro de uma CTE.

    A raiz da consulta é SELECT, mas a CTE contém UPDATE.
    A consulta deve ser rejeitada pela política de somente leitura.
    """

    sql = """
        WITH updated AS (
            UPDATE orders
            SET status = 'delivered'
            WHERE status = 'pending'
            RETURNING *
        )
        SELECT COUNT(*) AS total
        FROM updated
    """

    parsed_queries = sqlglot.parse(sql, read="duckdb")

    assert len(parsed_queries) == 1

    parsed_query = parsed_queries[0]

    assert isinstance(parsed_query, exp.Select)

    assert any(isinstance(node, exp.Update) for node in parsed_query.walk())

    with pytest.raises(SQLGuardrailError):
        validate_query(sql)


# ============================================================
# LIMITES DE TAMANHO E COMPLEXIDADE
# ============================================================


def test_rejects_sql_exceeding_maximum_length():
    """Rejeita consultas que ultrapassam 50.000 caracteres."""

    sql = "SELECT 1 FROM orders " + (" " * 50_001)

    with pytest.raises(
        SQLGuardrailError,
        match="comprimento máximo",
    ):
        validate_query(sql)


def test_rejects_ast_exceeding_maximum_node_count():
    """
    Rejeita consultas com mais de 2.000 nós na AST.

    Cada projeção adiciona múltiplos nós à estrutura sintática.
    """

    projections = ", ".join(f"{index} AS column_{index}" for index in range(700))

    sql = f"SELECT {projections} FROM orders"

    with pytest.raises(
        SQLGuardrailError,
        match="quantidade máxima de nós",
    ):
        validate_query(sql)


def test_rejects_ast_exceeding_maximum_depth():
    """Rejeita consultas com mais de 50 níveis de profundidade."""

    nested_expression = "1"

    for _ in range(55):
        nested_expression = f"({nested_expression})"

    sql = f"SELECT {nested_expression} FROM orders"

    with pytest.raises(
        SQLGuardrailError,
        match="profundidade máxima",
    ):
        validate_query(sql)


def test_rejects_excessive_number_of_ctes():
    """Rejeita consultas com mais de 20 CTEs."""

    ctes = [
        "cte_0 AS (SELECT id FROM orders)",
    ]

    for index in range(1, 21):
        ctes.append(f"cte_{index} AS (SELECT id FROM cte_{index - 1})")

    sql = "WITH " + ", ".join(ctes) + " SELECT id FROM cte_20"

    with pytest.raises(
        SQLGuardrailError,
        match="quantidade máxima de CTEs",
    ):
        validate_query(sql)


# ============================================================
# TESTES AVANÇADOS DE SEGURANÇA
# ============================================================


@pytest.mark.parametrize(
    "function_name",
    [
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_text",
        "read_blob",
        "parquet_scan",
        "csv_scan",
        "sqlite_scan",
        "postgres_scan",
    ],
)
def test_rejects_external_data_source_functions(function_name: str):
    """Bloqueia diferentes funções de leitura de fontes externas."""

    sql = f"SELECT * FROM {function_name}('external_resource')"

    with pytest.raises(
        SQLGuardrailError,
        match="Função bloqueada",
    ):
        validate_query(sql)


def test_rejects_blocked_function_case_insensitively():
    """Garante que alterar maiúsculas e minúsculas não contorna o bloqueio."""

    sql = "SELECT * FROM READ_CSV('external_data.csv')"

    with pytest.raises(
        SQLGuardrailError,
        match="Função bloqueada",
    ):
        validate_query(sql)


def test_rejects_external_function_inside_subquery():
    """Bloqueia funções perigosas mesmo dentro de subconsultas."""

    sql = """
        SELECT *
        FROM (
            SELECT *
            FROM read_parquet('external_data.parquet')
        ) AS source
    """

    with pytest.raises(
        SQLGuardrailError,
        match="Função bloqueada",
    ):
        validate_query(sql)


# ============================================================
# TESTES DE COMANDOS ADMINISTRATIVOS
# ============================================================


@pytest.mark.parametrize(
    "sql",
    [
        "COPY orders TO 'external_data.csv'",
        "INSTALL httpfs",
        "LOAD httpfs",
        "ATTACH 'external.duckdb' AS external_db",
        "DETACH external_db",
        "PRAGMA enable_external_access = false",
        "SET memory_limit = '16GB'",
        "DROP TABLE orders",
    ],
)
def test_rejects_administrative_commands(sql: str):
    """Rejeita comandos administrativos e operações de banco."""

    with pytest.raises(SQLGuardrailError):
        validate_query(sql)


# ============================================================
# TESTES DE CONTORNO POR CTE
# ============================================================


def test_rejects_unauthorized_table_hidden_inside_cte():
    """
    Garante que uma CTE não esconda uma tabela não autorizada.

    O nome da CTE é válido, mas sua origem não é autorizada.
    """

    sql = """
        WITH safe_data AS (
            SELECT *
            FROM confidential_data
        )
        SELECT *
        FROM safe_data
    """

    with pytest.raises(
        SQLGuardrailError,
        match="Tabela não autorizada",
    ):
        validate_query(sql)


def test_accepts_authorized_cte_with_multiple_references():
    """Permite referências legítimas repetidas a uma CTE autorizada."""

    sql = """
        WITH order_summary AS (
            SELECT status, COUNT(*) AS total
            FROM orders
            GROUP BY status
        )
        SELECT
            first_summary.status,
            first_summary.total,
            second_summary.total AS comparison_total
        FROM order_summary AS first_summary
        JOIN order_summary AS second_summary
            ON first_summary.status = second_summary.status
    """

    result = validate_query(sql)

    assert isinstance(result, str)
    assert "order_summary" in result.lower()


# ============================================================
# TESTES DE CATÁLOGOS EXTERNOS
# ============================================================


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM external_catalog.external_schema.orders",
        "SELECT * FROM external_schema.orders",
        'SELECT * FROM "external_schema"."orders"',
    ],
)
def test_rejects_external_catalog_and_schema_references(sql: str):
    """Bloqueia tabelas qualificadas por catálogo ou schema."""

    with pytest.raises(SQLGuardrailError):
        validate_query(sql)


# ============================================================
# TESTES DE ALLOWLIST DE FUNÇÕES SQL
# ============================================================


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) AS total FROM orders",
        "SELECT SUM(1) AS total FROM orders",
        "SELECT AVG(1) AS average FROM orders",
        "SELECT ROUND(1.234, 2) AS value FROM orders",
        "SELECT NULLIF(1, 0) AS value FROM orders",
        "SELECT STRFTIME(DATE '2025-01-01', '%Y-%m') AS month FROM orders",
        "SELECT HASH(1) AS value FROM orders",
    ],
)
def test_accepts_explicitly_allowed_functions(sql: str):
    """Preserva funções utilizadas pelo benchmark e testes internos."""

    result = validate_query(sql)

    assert isinstance(result, str)
    assert result.strip()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT CUSTOM_EXTERNAL_FUNCTION(1) FROM orders",
        "SELECT UNKNOWN_UDF(status) FROM orders",
    ],
)
def test_rejects_unknown_functions(sql: str):
    """Bloqueia funções não contempladas pela política de segurança."""

    with pytest.raises(
        SQLGuardrailError,
        match="Função não permitida",
    ):
        validate_query(sql)


# ============================================================
# TESTES DE COMPATIBILIDADE DA SINTAXE ANALÍTICA
# ============================================================


def test_accepts_filter_aggregation():
    """Preserva agregações condicionais FILTER."""

    sql = """
        SELECT
            COUNT(*) FILTER (WHERE status = 'canceled') AS canceled_orders
        FROM orders
    """

    result = validate_query(sql)

    assert "FILTER" in result.upper()


def test_accepts_case_expressions():
    """Preserva expressões condicionais CASE."""

    sql = """
        SELECT
            CASE
                WHEN status = 'canceled' THEN 'Cancelado'
                ELSE 'Outro'
            END AS order_status
        FROM orders
    """

    result = validate_query(sql)

    assert "CASE" in result.upper()


def test_accepts_scalar_subqueries():
    """Preserva subconsultas escalares utilizadas em análises compostas."""

    sql = """
        SELECT
            (SELECT COUNT(*) FROM orders) AS total_orders
    """

    result = validate_query(sql)

    assert "SELECT" in result.upper()
    assert "COUNT" in result.upper()


def test_accepts_joins_between_authorized_tables():
    """Preserva JOINs entre tabelas autorizadas."""

    sql = """
        SELECT o.*
        FROM orders AS o
        INNER JOIN restaurants AS r ON TRUE
    """

    result = validate_query(sql)

    assert "JOIN" in result.upper()
    assert "orders" in result.lower()
    assert "restaurants" in result.lower()
