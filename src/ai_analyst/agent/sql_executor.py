"""
Execução segura de consultas SQL no DuckDB.

Responsabilidades:
- Validar consultas antes da execução.
- Utilizar conexões somente leitura.
- Controlar o volume de resultados retornados.
- Estimar o consumo de memória dos resultados.
- Ler resultados incrementalmente.
- Identificar timeouts do DuckDB.
- Preservar o rastreamento das etapas de execução.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from sys import getsizeof
from time import perf_counter
from typing import Any

import duckdb

from ai_analyst.data.db import connect_readonly
from ai_analyst.security.guardrails import validate_query

# ==========================================================
# LIMITES DE EXECUÇÃO
# ==========================================================

MAX_RESULT_ROWS = 10_000

MAX_RESULT_BYTES = 10 * 1024 * 1024

FETCH_BATCH_SIZE = 128


# ==========================================================
# EXCEÇÕES
# ==========================================================


class SQLResultLimitError(RuntimeError):
    """Consulta ultrapassou os limites de retorno configurados."""


class SQLExecutionTimeoutError(TimeoutError):
    """Consulta ultrapassou o tempo máximo de execução do DuckDB."""


# ==========================================================
# EXECUTOR
# ==========================================================


class SQLExecutor:
    """Valida e executa consultas SQL no banco de dados.

    As consultas são validadas antes da abertura da conexão.

    Os resultados são consumidos incrementalmente, evitando
    carregar conjuntos ilimitados de registros na memória.

    Os limites de linhas e tamanho estimado podem ser
    configurados na inicialização.
    """

    def __init__(
        self,
        max_rows: int = MAX_RESULT_ROWS,
        max_result_bytes: int = MAX_RESULT_BYTES,
        fetch_batch_size: int = FETCH_BATCH_SIZE,
    ):
        self._validate_positive_integer(
            "max_rows",
            max_rows,
        )

        self._validate_positive_integer(
            "max_result_bytes",
            max_result_bytes,
        )

        self._validate_positive_integer(
            "fetch_batch_size",
            fetch_batch_size,
        )

        self.max_rows = max_rows
        self.max_result_bytes = max_result_bytes
        self.fetch_batch_size = fetch_batch_size

        self.stage_callback: Callable | None = None

    # ======================================================
    # VALIDAÇÃO DA CONFIGURAÇÃO
    # ======================================================

    @staticmethod
    def _validate_positive_integer(
        name: str,
        value: int,
    ) -> None:
        """Valida parâmetros inteiros estritamente positivos."""

        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} deve ser um inteiro positivo.")

    # ======================================================
    # CLASSIFICAÇÃO DE TIMEOUT
    # ======================================================

    @staticmethod
    def _is_execution_timeout(
        error: Exception,
    ) -> bool:
        """Identifica interrupções provocadas pelo timeout do DuckDB.

        Uma InterruptException isolada não é suficiente para
        classificar a falha como timeout.

        A mensagem também precisa indicar que o tempo máximo
        de execução foi excedido.
        """

        return (
            isinstance(error, duckdb.InterruptException)
            and "maximum execution time" in str(error).casefold()
        )

    # ======================================================
    # CALLBACKS DE EXECUÇÃO
    # ======================================================

    def set_stage_callback(
        self,
        callback: Callable | None,
    ) -> None:
        """Define um callback opcional para acompanhar as etapas."""

        if callback is not None and not callable(callback):
            raise TypeError("O callback de execução deve ser callable ou None.")

        self.stage_callback = callback

    def _notify_stage(
        self,
        stage: str,
        status: str,
        duration_ms: float,
    ) -> None:
        """Notifica o orquestrador sobre uma etapa da execução."""

        if self.stage_callback:
            self.stage_callback(
                stage,
                status,
                duration_ms,
            )

    # ======================================================
    # ESTIMATIVA DO TAMANHO DOS RESULTADOS
    # ======================================================

    @classmethod
    def _deep_sizeof(
        cls,
        value: Any,
        seen: set[int] | None = None,
    ) -> int:
        """Estima recursivamente o tamanho de um objeto em memória.

        Considera:
        - Tamanho do próprio objeto.
        - Chaves e valores de dicionários.
        - Elementos de listas, tuplas e conjuntos.
        - Proteção contra referências circulares.

        A estimativa utiliza sys.getsizeof e não representa
        um limite absoluto de memória do processo.
        """

        if seen is None:
            seen = set()

        object_id = id(value)

        if object_id in seen:
            return 0

        seen.add(object_id)

        total_size = getsizeof(value)

        if isinstance(value, Mapping):
            for key, item in value.items():
                total_size += cls._deep_sizeof(
                    key,
                    seen,
                )

                total_size += cls._deep_sizeof(
                    item,
                    seen,
                )

        elif isinstance(
            value,
            (list, tuple, set, frozenset),
        ):
            for item in value:
                total_size += cls._deep_sizeof(
                    item,
                    seen,
                )

        return total_size

    # ======================================================
    # LEITURA INCREMENTAL DOS RESULTADOS
    # ======================================================

    def _fetch_results(
        self,
        result: Any,
        columns: list[str],
    ) -> list[dict[str, Any]]:
        """Consome resultados incrementalmente e aplica limites.

        O executor interrompe a leitura quando:
        - A quantidade máxima de registros é ultrapassada.
        - O tamanho estimado dos resultados é ultrapassado.

        Nenhum resultado parcial é retornado em caso de excesso.
        """

        results: list[dict[str, Any]] = []

        estimated_size = 0

        while True:
            remaining_rows = self.max_rows - len(results)

            # Solicita no máximo uma linha além do limite.
            batch_size = min(
                self.fetch_batch_size,
                remaining_rows + 1,
            )

            rows = result.fetchmany(batch_size)

            if not rows:
                break

            for raw_row in rows:
                if len(results) >= self.max_rows:
                    raise SQLResultLimitError(
                        "A consulta ultrapassou o limite de "
                        f"{self.max_rows:,} registros."
                    )

                row = dict(
                    zip(
                        columns,
                        raw_row,
                        strict=True,
                    )
                )

                row_size = self._deep_sizeof(row)

                if estimated_size + row_size > self.max_result_bytes:
                    raise SQLResultLimitError(
                        "A consulta ultrapassou o limite estimado "
                        "de tamanho dos resultados "
                        f"({self.max_result_bytes:,} bytes)."
                    )

                results.append(row)

                estimated_size += row_size

        return results

    # ======================================================
    # EXECUÇÃO PRINCIPAL
    # ======================================================

    def execute(
        self,
        sql: str,
    ) -> list[dict[str, Any]]:
        """Valida e executa uma consulta SQL.

        Fluxo:
        1. Validação da consulta.
        2. Abertura de conexão somente leitura.
        3. Execução com timeout configurado no DuckDB.
        4. Leitura incremental dos resultados.
        5. Aplicação dos limites de linhas e tamanho.
        6. Classificação de timeout.
        7. Fechamento da conexão.

        Lança:
            SQLGuardrailError: Consulta inválida ou não autorizada.
            SQLResultLimitError: Limites de resultados ultrapassados.
            SQLExecutionTimeoutError: Tempo máximo de execução excedido.
            Exception: Outros erros operacionais do DuckDB.
        """

        # --------------------------------------------------
        # 1. VALIDAR A CONSULTA
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 2. EXECUTAR A CONSULTA
        # --------------------------------------------------

        stage_start = perf_counter()

        con = None

        try:
            con = connect_readonly()

            result = con.execute(validated_sql)

            description = result.description

            if description is None:
                raise RuntimeError("A consulta não retornou metadados de colunas.")

            columns = [column[0] for column in description]

            # --------------------------------------------------
            # 3. CONSUMIR RESULTADOS COM LIMITES
            # --------------------------------------------------

            results = self._fetch_results(
                result=result,
                columns=columns,
            )

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

        except Exception as exc:
            duration_ms = round(
                (perf_counter() - stage_start) * 1000,
                2,
            )

            self._notify_stage(
                "sql_execution",
                "failed",
                duration_ms,
            )

            if self._is_execution_timeout(exc):
                raise SQLExecutionTimeoutError(
                    "A consulta excedeu o tempo máximo "
                    "de execução configurado no DuckDB."
                ) from exc

            raise

        finally:
            if con is not None:
                con.close()
