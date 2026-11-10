"""
Execução segura de consultas SQL no DuckDB.
"""

from collections.abc import Callable
from time import perf_counter

from ai_analyst.data.db import connect_readonly
from ai_analyst.security.guardrails import validate_query


class SQLExecutor:
    """Valida e executa consultas SQL no banco de dados."""

    def __init__(self):
        self.stage_callback: Callable | None = None

    def set_stage_callback(self, callback: Callable | None):
        """Define um callback opcional para acompanhar as etapas."""

        self.stage_callback = callback

    def _notify_stage(
        self,
        stage: str,
        status: str,
        duration_ms: float,
    ):
        """Notifica o orquestrador sobre uma etapa da execução."""

        if self.stage_callback:
            self.stage_callback(
                stage,
                status,
                duration_ms,
            )

    def execute(self, sql: str) -> list[dict]:
        """
        Valida e executa uma consulta SQL.

        A validação sempre acontece antes da conexão com o banco.
        """

        # 1. Validar a consulta.
        stage_start = perf_counter()

        try:
            validated_sql = validate_query(sql)

        except Exception:
            duration_ms = round(
                (perf_counter() - stage_start) * 1000,
                2,
            )

            self._notify_stage(
                "sql_validation",
                "failed",
                duration_ms,
            )

            raise

        validation_ms = round(
            (perf_counter() - stage_start) * 1000,
            2,
        )

        self._notify_stage(
            "sql_validation",
            "completed",
            validation_ms,
        )

        # 2. Executar a consulta validada.
        stage_start = perf_counter()
        con = None

        try:
            con = connect_readonly()

            result = con.execute(validated_sql)

            columns = [description[0] for description in result.description]

            rows = result.fetchall()

            results = [dict(zip(columns, row)) for row in rows]

            duration_ms = round(
                (perf_counter() - stage_start) * 1000,
                2,
            )

            self._notify_stage(
                "sql_execution",
                "completed",
                duration_ms,
            )

            return results

        except Exception:
            duration_ms = round(
                (perf_counter() - stage_start) * 1000,
                2,
            )

            self._notify_stage(
                "sql_execution",
                "failed",
                duration_ms,
            )

            raise

        finally:
            if con is not None:
                con.close()
