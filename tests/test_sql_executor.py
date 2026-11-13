from __future__ import annotations

import duckdb
import pytest

import ai_analyst.agent.sql_executor as sql_executor_module
import ai_analyst.data.db as db_module
from ai_analyst.agent.sql_executor import (
    SQLExecutionTimeoutError,
    SQLExecutor,
    SQLResultLimitError,
)
from ai_analyst.data.db import build_database
from ai_analyst.security.guardrails import SQLGuardrailError

# ==========================================================
# FIXTURES
# ==========================================================


@pytest.fixture(scope="module")
def executor():
    build_database()
    return SQLExecutor()


# ==========================================================
# EXECUÇÃO BÁSICA
# ==========================================================


def test_execute_valid_query(executor):
    results = executor.execute(
        """
        SELECT city, COUNT(*) AS total
        FROM orders
        GROUP BY city
        ORDER BY total DESC
        LIMIT 5
        """
    )

    assert len(results) == 5
    assert "city" in results[0]
    assert "total" in results[0]


def test_reject_unsafe_query(executor):
    with pytest.raises(SQLGuardrailError):
        executor.execute("SELECT * FROM read_csv('arquivo.csv')")


# ==========================================================
# LIMITES DE LINHAS
# ==========================================================


def test_execute_accepts_exact_row_limit():
    query_executor = SQLExecutor(
        max_rows=5,
        fetch_batch_size=2,
    )

    results = query_executor.execute(
        """
        SELECT order_id
        FROM orders
        ORDER BY order_id
        LIMIT 5
        """
    )

    assert len(results) == 5
    assert results[0]["order_id"] == 1
    assert results[-1]["order_id"] == 5


def test_execute_rejects_rows_above_limit():
    query_executor = SQLExecutor(
        max_rows=5,
        fetch_batch_size=2,
    )

    with pytest.raises(
        SQLResultLimitError,
        match="limite de 5 registros",
    ):
        query_executor.execute(
            """
            SELECT order_id
            FROM orders
            ORDER BY order_id
            LIMIT 6
            """
        )


def test_execute_accepts_results_below_row_limit():
    query_executor = SQLExecutor(
        max_rows=10,
    )

    results = query_executor.execute(
        """
        SELECT order_id
        FROM orders
        ORDER BY order_id
        LIMIT 3
        """
    )

    assert len(results) == 3


# ==========================================================
# LIMITES DE TAMANHO DOS RESULTADOS
# ==========================================================


def test_execute_accepts_result_within_memory_limit():
    query_executor = SQLExecutor(
        max_result_bytes=10_000,
    )

    results = query_executor.execute(
        """
        SELECT order_id
        FROM orders
        LIMIT 1
        """
    )

    assert len(results) == 1


def test_execute_rejects_result_above_memory_limit():
    query_executor = SQLExecutor(
        max_result_bytes=1,
    )

    with pytest.raises(
        SQLResultLimitError,
        match="limite estimado",
    ):
        query_executor.execute(
            """
            SELECT order_id
            FROM orders
            LIMIT 1
            """
        )


# ==========================================================
# VALIDAÇÃO DOS PARÂMETROS
# ==========================================================


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"max_rows": 0}, "max_rows"),
        ({"max_rows": -1}, "max_rows"),
        ({"max_rows": True}, "max_rows"),
        ({"max_result_bytes": 0}, "max_result_bytes"),
        ({"max_result_bytes": -1}, "max_result_bytes"),
        ({"max_result_bytes": False}, "max_result_bytes"),
        ({"fetch_batch_size": 0}, "fetch_batch_size"),
        ({"fetch_batch_size": -1}, "fetch_batch_size"),
        ({"fetch_batch_size": True}, "fetch_batch_size"),
    ],
)
def test_constructor_rejects_invalid_limits(kwargs, parameter):
    with pytest.raises(
        ValueError,
        match=parameter,
    ):
        SQLExecutor(**kwargs)


def test_constructor_accepts_custom_limits():
    query_executor = SQLExecutor(
        max_rows=500,
        max_result_bytes=2_000_000,
        fetch_batch_size=64,
    )

    assert query_executor.max_rows == 500
    assert query_executor.max_result_bytes == 2_000_000
    assert query_executor.fetch_batch_size == 64


# ==========================================================
# ESTIMATIVA DE MEMÓRIA
# ==========================================================


def test_deep_sizeof_counts_nested_objects():
    value = {
        "name": "AI Business Analyst",
        "metrics": {
            "orders": 60_000,
            "rate": 3.62,
        },
    }

    size = SQLExecutor._deep_sizeof(value)

    assert isinstance(size, int)
    assert size > 0


def test_deep_sizeof_handles_repeated_references():
    from sys import getsizeof

    shared_value = ["shared", 123]

    value = [
        shared_value,
        shared_value,
    ]

    size = SQLExecutor._deep_sizeof(value)

    expected_size = getsizeof(value) + SQLExecutor._deep_sizeof(shared_value)

    assert size == expected_size


# ==========================================================
# CALLBACKS DE EXECUÇÃO
# ==========================================================


def test_stage_callback_receives_successful_stages():
    query_executor = SQLExecutor()

    events = []

    query_executor.set_stage_callback(
        lambda stage, status, duration: events.append((stage, status, duration))
    )

    results = query_executor.execute("SELECT COUNT(*) AS total FROM orders")

    assert len(results) == 1

    assert [(stage, status) for stage, status, _ in events] == [
        ("sql_validation", "completed"),
        ("sql_execution", "completed"),
    ]

    assert all(
        isinstance(duration, float) and duration >= 0 for _, _, duration in events
    )


def test_stage_callback_receives_execution_failure():
    query_executor = SQLExecutor(
        max_rows=1,
    )

    events = []

    query_executor.set_stage_callback(
        lambda stage, status, duration: events.append((stage, status, duration))
    )

    with pytest.raises(SQLResultLimitError):
        query_executor.execute(
            """
            SELECT order_id
            FROM orders
            ORDER BY order_id
            LIMIT 2
            """
        )

    assert [(stage, status) for stage, status, _ in events] == [
        ("sql_validation", "completed"),
        ("sql_execution", "failed"),
    ]


def test_stage_callback_receives_validation_failure():
    query_executor = SQLExecutor()

    events = []

    query_executor.set_stage_callback(
        lambda stage, status, duration: events.append((stage, status, duration))
    )

    with pytest.raises(SQLGuardrailError):
        query_executor.execute("SELECT * FROM read_csv('arquivo.csv')")

    assert len(events) == 1
    assert events[0][0] == "sql_validation"
    assert events[0][1] == "failed"


def test_set_stage_callback_rejects_non_callable():
    query_executor = SQLExecutor()

    with pytest.raises(
        TypeError,
        match="callable",
    ):
        query_executor.set_stage_callback("invalid")


def test_set_stage_callback_accepts_none():
    query_executor = SQLExecutor()

    query_executor.set_stage_callback(None)

    assert query_executor.stage_callback is None


# ==========================================================
# CLASSIFICAÇÃO DE TIMEOUT
# ==========================================================


def test_interrupt_without_execution_timeout_is_not_classified():
    error = duckdb.InterruptException("INTERRUPT Error: Query was interrupted.")

    assert SQLExecutor._is_execution_timeout(error) is False


def test_non_duckdb_exception_is_not_classified_as_timeout():
    error = RuntimeError("Query exceeded maximum execution time")

    assert SQLExecutor._is_execution_timeout(error) is False


# ==========================================================
# TIMEOUT REAL DO DUCKDB
# ==========================================================


def test_execute_converts_duckdb_timeout_and_closes_connection(
    monkeypatch,
):
    """
    Verifica o timeout real do DuckDB.

    Também garante:
    - Conversão para SQLExecutionTimeoutError.
    - Preservação da exceção original como causa.
    - Callback de falha.
    - Fechamento da conexão.
    """

    # Reduz o timeout somente para esta execução de teste.
    monkeypatch.setattr(
        db_module,
        "DUCKDB_MAX_EXECUTION_TIME_MS",
        1,
    )

    # Cria uma conexão real com timeout de 1 ms.
    real_connection = db_module.connect_readonly()

    connection_state = {
        "closed": False,
    }

    class ConnectionSpy:
        """Observa o fechamento da conexão real."""

        def execute(self, query):
            return real_connection.execute(query)

        def close(self):
            connection_state["closed"] = True
            real_connection.close()

    connection_spy = ConnectionSpy()

    # Injeta a conexão rastreável no SQLExecutor.
    monkeypatch.setattr(
        sql_executor_module,
        "connect_readonly",
        lambda: connection_spy,
    )

    query_executor = SQLExecutor()

    events = []

    query_executor.set_stage_callback(
        lambda stage, status, duration: events.append((stage, status, duration))
    )

    expensive_query = """
        SELECT SUM(hash(a.order_id + b.order_id)) AS result
        FROM orders AS a
        CROSS JOIN orders AS b
    """

    try:
        with pytest.raises(
            SQLExecutionTimeoutError,
            match="tempo máximo",
        ) as exc_info:
            query_executor.execute(expensive_query)

        # A exceção original do DuckDB deve ser preservada.
        assert isinstance(
            exc_info.value.__cause__,
            duckdb.InterruptException,
        )

        assert "maximum execution time" in str(exc_info.value.__cause__).casefold()

        # A conexão deve estar fechada após a interrupção.
        assert connection_state["closed"] is True

        # A validação SQL passou, mas a execução falhou.
        assert [(stage, status) for stage, status, _ in events] == [
            ("sql_validation", "completed"),
            ("sql_execution", "failed"),
        ]

        assert all(duration >= 0 for _, _, duration in events)

    finally:
        # Proteção adicional para evitar deixar a conexão aberta
        # mesmo se alguma asserção do teste falhar.
        if not connection_state["closed"]:
            connection_spy.close()


# ==========================================================
# GERENCIAMENTO DE CONEXÃO
# ==========================================================


def test_connection_is_closed_when_result_limit_is_exceeded(
    monkeypatch,
):
    real_connection = sql_executor_module.connect_readonly()

    connection_state = {
        "closed": False,
    }

    class ConnectionSpy:
        """Rastreia o fechamento da conexão real."""

        def execute(self, query):
            return real_connection.execute(query)

        def close(self):
            connection_state["closed"] = True
            real_connection.close()

    connection_spy = ConnectionSpy()

    monkeypatch.setattr(
        sql_executor_module,
        "connect_readonly",
        lambda: connection_spy,
    )

    query_executor = SQLExecutor(
        max_rows=1,
    )

    with pytest.raises(SQLResultLimitError):
        query_executor.execute(
            """
            SELECT order_id
            FROM orders
            ORDER BY order_id
            LIMIT 2
            """
        )

    assert connection_state["closed"] is True


# ==========================================================
# INTEGRAÇÃO REAL: ALLOWLIST E DUCKDB
# ==========================================================


@pytest.mark.parametrize(
    ("sql", "expected_column"),
    [
        (
            """
            SELECT STRFTIME(order_ts, '%Y-%m') AS order_month
            FROM orders
            LIMIT 1
            """,
            "order_month",
        ),
        (
            """
            SELECT CAST(DATE '2025-01-01' AS VARCHAR) AS date_value
            FROM orders
            LIMIT 1
            """,
            "date_value",
        ),
        (
            """
            SELECT
                CASE
                    WHEN status = 'canceled' THEN 'canceled'
                    ELSE 'other'
                END AS status_group
            FROM orders
            LIMIT 1
            """,
            "status_group",
        ),
        (
            """
            SELECT
                IF(status = 'canceled', 'canceled', 'other') AS status_group
            FROM orders
            LIMIT 1
            """,
            "status_group",
        ),
        (
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE status = 'canceled'
                ) AS canceled_orders
            FROM orders
            """,
            "canceled_orders",
        ),
        (
            """
            SELECT
                ROUND(SUM(items_total - discount), 2) AS net_items_value
            FROM orders
            """,
            "net_items_value",
        ),
        (
            """
            SELECT
                ROUND(AVG(delivery_minutes), 2) AS average_delivery_minutes
            FROM orders
            WHERE status = 'delivered'
              AND delivery_minutes IS NOT NULL
            """,
            "average_delivery_minutes",
        ),
        (
            """
            SELECT
                ROUND(
                    COUNT(*) FILTER (WHERE status = 'canceled')
                    * 100.0 / NULLIF(COUNT(*), 0),
                    2
                ) AS cancellation_rate_pct
            FROM orders
            """,
            "cancellation_rate_pct",
        ),
        (
            """
            SELECT
                order_id,
                ROW_NUMBER() OVER (ORDER BY order_id) AS row_number
            FROM orders
            LIMIT 3
            """,
            "row_number",
        ),
        (
            """
            SELECT
                (SELECT COUNT(*) FROM orders) AS total_orders
            """,
            "total_orders",
        ),
        (
            """
            SELECT
                r.category,
                COUNT(*) AS order_count
            FROM orders AS o
            INNER JOIN restaurants AS r
                ON o.restaurant_id = r.restaurant_id
            GROUP BY r.category
            ORDER BY r.category
            LIMIT 3
            """,
            "category",
        ),
    ],
    ids=[
        "strftime",
        "date-and-cast",
        "case",
        "if",
        "filter-aggregation",
        "sum",
        "avg-and-round",
        "nullif-and-percentage",
        "window-function",
        "scalar-subquery",
        "authorized-join",
    ],
)
def test_execute_allowed_analytical_sql_features(
    executor,
    sql: str,
    expected_column: str,
):
    """
    Verifica a execução real das expressões autorizadas.

    As consultas passam pelo SQLGuardrail e são executadas
    no DuckDB local, sem chamadas a provedores de IA.
    """

    results = executor.execute(sql)

    assert results
    assert expected_column in results[0]
