"""Criacao do banco DuckDB e conexao SOMENTE LEITURA para o agente.

Seguranca: o agente executara SQL gerado por LLM. Por isso a conexao e
read-only, sem acesso a arquivos externos (read_csv, COPY, ATTACH...) e com
a configuracao travada, para o SQL nao conseguir reabilitar nada.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from .generate import generate_tables

DEFAULT_DB_PATH = Path("data") / "delivery.duckdb"

def get_database_version(
    path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Gera uma versão baseada nas propriedades do arquivo do banco."""

    db_path = Path(path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Banco de dados não encontrado: {db_path}"
        )

    stats = db_path.stat()

    return f"{stats.st_mtime_ns}-{stats.st_size}"

def build_database(path: str | Path = DEFAULT_DB_PATH, seed: int = 42,
                   n_orders: int = 60_000) -> Path:
    """Gera os dados sinteticos e grava o banco (recria se ja existir)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    tables = generate_tables(seed=seed, n_orders=n_orders)
    con = duckdb.connect(str(path))
    try:
        for name, df in tables.items():  # nomes sao constantes internas
            con.register("_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")
    finally:
        con.close()
    return path


def connect_readonly(path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Conexao segura para executar SQL vindo do LLM."""
    con = duckdb.connect(str(path), read_only=True,
                         config={"enable_external_access": False})
    con.execute("SET lock_configuration = true")
    return con


if __name__ == "__main__":
    db = build_database()
    con = connect_readonly(db)
    print(f"Banco criado em {db}")
    for t in ("restaurants", "customers", "couriers", "orders"):
        print(f"  {t:12s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>7,} linhas")
    con.close()
