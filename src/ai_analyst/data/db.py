"""Criação do banco DuckDB e conexão SOMENTE LEITURA para o agente.

Segurança:
- Conexão read-only.
- Sem acesso a arquivos ou recursos externos.
- Configuração bloqueada após inicialização.
- Limite de memória gerenciada pelo DuckDB.
- Limite de armazenamento temporário.
- Limite de threads.
- Tempo máximo de execução por consulta.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .generate import generate_tables

DEFAULT_DB_PATH = Path("data") / "delivery.duckdb"


# ==========================================================
# LIMITES DE RECURSOS DO DUCKDB
# ==========================================================

DUCKDB_MEMORY_LIMIT = "1GB"

DUCKDB_MAX_TEMP_DIRECTORY_SIZE = "1GB"

DUCKDB_THREADS = 4

DUCKDB_MAX_EXECUTION_TIME_MS = 30_000


# ==========================================================
# VERSÃO DO BANCO
# ==========================================================


def get_database_version(
    path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Gera uma versão baseada nas propriedades do arquivo do banco."""

    db_path = Path(path)

    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")

    stats = db_path.stat()

    return f"{stats.st_mtime_ns}-{stats.st_size}"


# ==========================================================
# CONSTRUÇÃO DO BANCO
# ==========================================================


def build_database(
    path: str | Path = DEFAULT_DB_PATH,
    seed: int = 42,
    n_orders: int = 60_000,
) -> Path:
    """Gera os dados sintéticos e grava o banco.

    O banco existente é recriado.

    Esta operação é destinada ao processo controlado de
    construção do dataset, não à execução de SQL do agente.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        path.unlink()

    tables = generate_tables(
        seed=seed,
        n_orders=n_orders,
    )

    con = duckdb.connect(str(path))

    try:
        for name, dataframe in tables.items():
            con.register(
                "_df",
                dataframe,
            )

            try:
                con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _df').fetchall()

            finally:
                con.unregister("_df")

    finally:
        con.close()

    return path


# ==========================================================
# CONEXÃO SEGURA SOMENTE LEITURA
# ==========================================================


def connect_readonly(
    path: str | Path = DEFAULT_DB_PATH,
) -> duckdb.DuckDBPyConnection:
    """Cria uma conexão restrita para executar SQL do agente.

    Configurações aplicadas:

    - read_only: impede modificações no banco.
    - enable_external_access: bloqueia acesso externo.
    - memory_limit: limita memória gerenciada pelo DuckDB.
    - max_temp_directory_size: limita arquivos temporários.
    - threads: restringe paralelismo.
    - max_execution_time: interrompe consultas demoradas.
    - lock_configuration: impede alterações posteriores.

    A configuração de timeout é aplicada antes do bloqueio
    definitivo das configurações.

    Se a configuração falhar, a conexão é fechada antes
    de propagar a exceção.
    """

    db_path = Path(path)

    if not db_path.exists():
        raise FileNotFoundError(f"Banco de dados não encontrado: {db_path}")

    connection_config = {
        "enable_external_access": False,
        "memory_limit": DUCKDB_MEMORY_LIMIT,
        "max_temp_directory_size": DUCKDB_MAX_TEMP_DIRECTORY_SIZE,
        "threads": DUCKDB_THREADS,
    }

    con = duckdb.connect(
        str(db_path),
        read_only=True,
        config=connection_config,
    )

    try:
        # --------------------------------------------------
        # TIMEOUT POR CONSULTA
        # --------------------------------------------------

        con.execute(
            f"SET max_execution_time = {DUCKDB_MAX_EXECUTION_TIME_MS}"
        ).fetchall()

        # --------------------------------------------------
        # BLOQUEIO DE ALTERAÇÃO DAS CONFIGURAÇÕES
        # --------------------------------------------------

        con.execute("SET lock_configuration = true").fetchall()

        return con

    except Exception:
        con.close()
        raise


# ==========================================================
# EXECUÇÃO DIRETA PARA CONSTRUÇÃO E INSPEÇÃO DO BANCO
# ==========================================================


if __name__ == "__main__":
    db = build_database()

    con = connect_readonly(db)

    try:
        print(f"Banco criado em {db}")

        print("\nConfigurações de segurança:")

        settings = con.execute(
            """
            SELECT name, value
            FROM duckdb_settings()
            WHERE name IN (
                'memory_limit',
                'max_temp_directory_size',
                'threads',
                'max_execution_time',
                'enable_external_access',
                'lock_configuration'
            )
            ORDER BY name
            """
        ).fetchall()

        for name, value in settings:
            print(f"  {name:28s} {value}")

        print("\nQuantidade de registros:")

        for table in (
            "restaurants",
            "customers",
            "couriers",
            "orders",
        ):
            count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchall()[0][0]

            print(f"  {table:12s} {count:>7,} linhas")

    finally:
        con.close()
