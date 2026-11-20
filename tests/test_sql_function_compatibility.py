from __future__ import annotations

from datetime import datetime

import pytest

from ai_analyst.data.db import build_database, connect_readonly
from ai_analyst.security.guardrails import (
    ALLOWED_FUNCTIONS,
    validate_query,
)

# ============================================================
# AUDITORIA DA ALLOWLIST
# ============================================================


# Entradas que não precisam aparecer como funções convencionais
# em duckdb_functions(), por serem construções especiais,
# aliases ou representações internas do SQLGlot.
DOCUMENTED_NON_CATALOG_FUNCTIONS = frozenset(
    {
        "cast",
        "try_cast",
        "date",
        "coalesce",
        "ifnull",
        "nvl",
        "str_to_time",
        "time_to_str",
    }
)


# Nomes internos utilizados pelo SQLGlot e seus equivalentes
# efetivos no DuckDB.
SQLGLOT_FUNCTION_ALIASES = {
    "str_to_time": "strptime",
    "time_to_str": "strftime",
}


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture(scope="module")
def connection():
    """Disponibiliza uma conexão real com o DuckDB."""

    build_database()
    con = connect_readonly()

    yield con

    con.close()


# ============================================================
# EXECUÇÃO REAL DE FUNÇÕES ESPECIAIS
# ============================================================


@pytest.mark.parametrize(
    ("function_name", "sql", "expected"),
    [
        (
            "COALESCE",
            "SELECT COALESCE(NULL, 'fallback') AS result",
            ("fallback",),
        ),
        (
            "IFNULL",
            "SELECT IFNULL(NULL, 'fallback') AS result",
            ("fallback",),
        ),
        (
            "NVL",
            "SELECT NVL(NULL, 'fallback') AS result",
            ("fallback",),
        ),
        (
            "STRPTIME",
            "SELECT STRPTIME('2025-01-31', '%Y-%m-%d') AS result",
            (datetime(2025, 1, 31),),
        ),
    ],
    ids=["coalesce", "ifnull", "nvl", "strptime"],
)
def test_allowed_functions_execute_in_duckdb(
    connection,
    function_name,
    sql,
    expected,
):
    """Confirma que funções autorizadas passam pelo guardrail e executam."""

    validated_sql = validate_query(sql)
    result = connection.execute(validated_sql).fetchone()

    assert result == expected, f"Falha na função {function_name}"


# ============================================================
# AUDITORIA DO CATÁLOGO DE FUNÇÕES
# ============================================================


def test_all_allowed_functions_are_accounted_for_in_duckdb_catalog(
    connection,
):
    """
    Garante que todas as entradas da allowlist sejam identificadas.

    Cada entrada deve:
    - Existir no catálogo de funções do DuckDB; ou
    - Estar documentada como construção especial, alias
      ou representação interna do SQLGlot.
    """

    rows = connection.execute(
        """
        SELECT DISTINCT lower(function_name)
        FROM duckdb_functions()
        """
    ).fetchall()

    available_functions = {row[0] for row in rows}

    unclassified_functions = (
        ALLOWED_FUNCTIONS - available_functions - DOCUMENTED_NON_CATALOG_FUNCTIONS
    )

    assert not unclassified_functions, (
        "Existem funções autorizadas sem classificação "
        "no catálogo do DuckDB: "
        f"{sorted(unclassified_functions)}"
    )

    # Evita manter exceções documentadas que não fazem
    # efetivamente parte da política de autorização.
    assert DOCUMENTED_NON_CATALOG_FUNCTIONS <= ALLOWED_FUNCTIONS


# ============================================================
# COMPATIBILIDADE ENTRE SQLGLOT E DUCKDB
# ============================================================


def test_sqlglot_function_aliases_map_to_duckdb_functions(
    connection,
):
    """
    Verifica os nomes internos do SQLGlot e seus equivalentes DuckDB.

    Esses nomes internos não devem ser confundidos com funções
    SQL adicionais que precisariam ser registradas no banco.
    """

    rows = connection.execute(
        """
        SELECT DISTINCT lower(function_name)
        FROM duckdb_functions()
        """
    ).fetchall()

    available_functions = {row[0] for row in rows}

    for internal_name, duckdb_name in SQLGLOT_FUNCTION_ALIASES.items():
        assert internal_name in ALLOWED_FUNCTIONS

        assert duckdb_name in available_functions, (
            f"O nome interno {internal_name} aponta para "
            f"{duckdb_name}, que não foi encontrado no catálogo DuckDB."
        )


# ============================================================
# EXECUÇÃO DE CONVERSÕES DE TIPO
# ============================================================


def test_try_cast_executes_in_duckdb(connection):
    """Confirma a execução real de TRY_CAST."""

    sql = """
        SELECT TRY_CAST('not-a-number' AS INTEGER) AS result
    """

    validated_sql = validate_query(sql)
    result = connection.execute(validated_sql).fetchone()

    assert result == (None,)
