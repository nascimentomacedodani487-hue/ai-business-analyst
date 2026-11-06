
import sqlglot
from sqlglot import exp


class SQLGuardrailError(Exception):
    """Base exception for SQL security violations."""


class SQLSyntaxError(SQLGuardrailError):
    """SQL syntax error eligible for controlled correction."""

    def __init__(self, sql: str, message: str):
        self.sql = sql
        super().__init__(message)


# Funções que podem acessar arquivos ou recursos externos.
BLOCKED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_ndjson",
    "read_text",
    "read_blob",
    "write_parquet",
    "write_csv",
    "write_json",
    "write_text",
    "parquet_scan",
    "csv_scan",
    "sqlite_scan",
    "postgres_scan",
    "glob",
    "httpfs",
    "query",
    "query_table",
    "query_table_by_name",
    "execute",
    "pragma",
    "install",
    "load",
}


# Tabelas que o agente está autorizado a consultar.
ALLOWED_TABLES = {
    "orders",
    "restaurants",
    "customers",
    "couriers",
}


def validate_query(sql_string: str) -> str:
    """
    Valida uma query SQL antes de permitir sua execução.

    Regras:
    1. Aceita apenas comandos SELECT.
    2. Bloqueia múltiplas instruções.
    3. Bloqueia funções que acessam arquivos ou recursos externos.
    4. Permite somente tabelas autorizadas ou CTEs declaradas.
    5. Bloqueia schemas e catálogos externos.
    6. Utiliza SQLGlot para analisar a AST.

    Retorna:
        Query SQL normalizada.

    Lança:
        SQLGuardrailError: Quando uma regra de segurança é violada.
    """

    # 1. Verificar entrada vazia ou inválida.
    if not isinstance(sql_string, str) or not sql_string.strip():
        raise SQLGuardrailError(
            "A query não pode estar vazia."
        )

    # 2. Interpretar o SQL utilizando o dialeto DuckDB.
    try:
        parsed_queries = sqlglot.parse(
            sql_string,
            read="duckdb"
        )

    except sqlglot.errors.ParseError as exc:
        raise SQLSyntaxError(
            sql_string,
            f"Sintaxe SQL invalida: {exc}",
        ) from exc

    # 3. Remover elementos vazios.
    parsed_queries = [
        query for query in parsed_queries
        if query is not None
    ]

    # 4. Bloquear múltiplas instruções.
    if len(parsed_queries) != 1:
        raise SQLGuardrailError(
            "Múltiplas instruções SQL não são permitidas."
        )

    query = parsed_queries[0]

    # 5. Permitir somente SELECT como comando raiz.
    if not isinstance(query, exp.Select):
        raise SQLGuardrailError(
            "Somente instruções SELECT são permitidas."
        )

    # 6. Identificar as CTEs declaradas.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in query.find_all(exp.CTE)
        if cte.alias_or_name
    }

    # 7. Percorrer a AST procurando funções e tabelas proibidas.
    for node in query.walk():

        # Bloquear funções que acessam arquivos ou recursos externos.
        if isinstance(node, exp.Func):
            function_names = {
                node.name.lower(),
                node.sql_name().lower(),
            }

            if isinstance(node, exp.Anonymous):
                function_names.add(node.name.lower())

            blocked = function_names.intersection(
                BLOCKED_FUNCTIONS
            )

            if blocked:
                function_name = sorted(blocked)[0]

                raise SQLGuardrailError(
                    f"Função bloqueada: {function_name}"
                )

        # Validar todas as tabelas utilizadas na consulta.
        if isinstance(node, exp.Table):

            # Identificar funções utilizadas como fontes de tabela.
            table_source = node.this

            if isinstance(table_source, exp.Func):
                function_names = {
                    table_source.sql_name().lower(),
                }

                if isinstance(table_source, exp.Anonymous):
                    function_names.add(
                        table_source.name.lower()
                    )

                blocked = function_names.intersection(
                    BLOCKED_FUNCTIONS
                )

                if blocked:
                    function_name = sorted(blocked)[0]

                    raise SQLGuardrailError(
                        f"Função bloqueada: {function_name}"
                    )

            table_name = node.name.lower()

            # Bloquear tabelas de schemas ou catálogos externos.
            if node.db or node.catalog:
                raise SQLGuardrailError(
                    f"Tabela não autorizada: {node.sql()}"
                )

            # Permitir referências às CTEs declaradas.
            if table_name in cte_names:
                continue

            # Bloquear referências sem nome de tabela válido.
            if not table_name:
                raise SQLGuardrailError(
                    "Fonte de tabela não autorizada."
                )

            # Verificar se a tabela física está autorizada.
            if table_name not in ALLOWED_TABLES:
                raise SQLGuardrailError(
                    f"Tabela não autorizada: {table_name}"
                )

    # 8. Retornar SQL normalizado.
    return query.sql(dialect="duckdb")
