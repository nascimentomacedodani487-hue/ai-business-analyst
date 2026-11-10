import pytest

from ai_analyst.agent.sql_executor import SQLExecutor
from ai_analyst.data.db import build_database
from ai_analyst.security.guardrails import SQLGuardrailError


@pytest.fixture(scope="module")
def executor():
    build_database()
    return SQLExecutor()


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
