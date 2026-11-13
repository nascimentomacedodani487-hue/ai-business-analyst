"""Validação de segurança de consultas SQL utilizando SQLGlot."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

# ============================================================
# LIMITES DE SEGURANÇA
# ============================================================

MAX_SQL_LENGTH = 50_000
MAX_AST_NODES = 2_000
MAX_AST_DEPTH = 50
MAX_CTES = 20


# ============================================================
# EXCEÇÕES
# ============================================================


class SQLGuardrailError(Exception):
    """Exceção base para violações de segurança SQL."""


class SQLSyntaxError(SQLGuardrailError):
    """Erro de sintaxe SQL elegível para correção controlada."""

    def __init__(self, sql: str, message: str):
        self.sql = sql
        super().__init__(message)


# ============================================================
# POLÍTICAS DE FUNÇÕES SQL
# ============================================================


ALLOWED_FUNCTIONS = frozenset(
    {
        # Agregações.
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "median",
        "stddev",
        "stddev_pop",
        "stddev_samp",
        "variance",
        "var_pop",
        "var_samp",
        "quantile_cont",
        "quantile_disc",
        "approx_quantile",
        "corr",
        "covar_pop",
        "covar_samp",
        # Matemática.
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "ceiling",
        "cos",
        "degrees",
        "exp",
        "floor",
        "greatest",
        "least",
        "ln",
        "log",
        "log10",
        "pi",
        "power",
        "pow",
        "radians",
        "sign",
        "sin",
        "sqrt",
        "tan",
        "trunc",
        "round",
        # Conversões de tipos.
        "cast",
        "try_cast",
        "date",
        # Controle de valores ausentes.
        "coalesce",
        "ifnull",
        "nvl",
        "nullif",
        # Texto.
        "lower",
        "upper",
        "trim",
        "ltrim",
        "rtrim",
        "length",
        "char_length",
        "character_length",
        "substring",
        "substr",
        "concat",
        "concat_ws",
        "replace",
        "split_part",
        "regexp_replace",
        "regexp_matches",
        "starts_with",
        "ends_with",
        # Data e hora.
        "strftime",
        "time_to_str",
        "strptime",
        "str_to_time",
        "date_trunc",
        "date_part",
        "date_diff",
        "datediff",
        "date_sub",
        "date_add",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "dayofweek",
        "dayofyear",
        "week",
        "quarter",
        # Funções de janela.
        "row_number",
        "rank",
        "dense_rank",
        "percent_rank",
        "cume_dist",
        "ntile",
        "lag",
        "lead",
        "first_value",
        "last_value",
        "nth_value",
        # Hash e integridade.
        "hash",
        "md5",
        "sha256",
    }
)


# Construções sintáticas representadas pelo SQLGlot como exp.Func,
# mas que não representam chamadas convencionais de funções.
SAFE_FUNCTION_EXPRESSIONS = frozenset(
    {
        "Case",
        "If",
        "And",
        "Or",
        "Not",
    }
)


# Defesa adicional para funções comprovadamente perigosas.
BLOCKED_FUNCTIONS = frozenset(
    {
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
)


# ============================================================
# POLÍTICAS DE COMANDOS
# ============================================================


BLOCKED_STATEMENT_NODES = frozenset(
    {
        "Alter",
        "Attach",
        "Command",
        "Commit",
        "Copy",
        "Create",
        "Delete",
        "Detach",
        "Drop",
        "Execute",
        "Export",
        "Grant",
        "Import",
        "Insert",
        "Install",
        "Into",
        "Load",
        "Merge",
        "Pragma",
        "Refresh",
        "Revoke",
        "Rollback",
        "Set",
        "SetItem",
        "Transaction",
        "Truncate",
        "TruncateTable",
        "Update",
        "Use",
    }
)


# ============================================================
# TABELAS AUTORIZADAS
# ============================================================


ALLOWED_TABLES = frozenset(
    {
        "orders",
        "restaurants",
        "customers",
        "couriers",
    }
)


# ============================================================
# PROTEÇÃO ANTES DO PARSING
# ============================================================


def _validate_sql_length(sql: str) -> None:
    """Impede entradas SQL excessivamente grandes."""

    if len(sql) > MAX_SQL_LENGTH:
        raise SQLGuardrailError("A consulta excede o comprimento máximo permitido.")


def _validate_parenthesis_depth(sql: str) -> None:
    """
    Limita o aninhamento de parênteses antes do parsing.

    Ignora parênteses dentro de strings, identificadores
    entre aspas e comentários SQL.
    """

    depth = 0
    index = 0
    length = len(sql)

    quote_char: str | None = None
    line_comment = False
    block_comment_depth = 0

    while index < length:
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < length else ""

        if line_comment:
            if char in "\r\n":
                line_comment = False

            index += 1
            continue

        if block_comment_depth:
            if char == "/" and next_char == "*":
                block_comment_depth += 1
                index += 2
                continue

            if char == "*" and next_char == "/":
                block_comment_depth -= 1
                index += 2
                continue

            index += 1
            continue

        if quote_char is not None:
            if char == "\\" and index + 1 < length:
                index += 2
                continue

            if char == quote_char:
                if next_char == quote_char:
                    index += 2
                    continue

                quote_char = None

            index += 1
            continue

        if char == "-" and next_char == "-":
            line_comment = True
            index += 2
            continue

        if char == "/" and next_char == "*":
            block_comment_depth = 1
            index += 2
            continue

        if char in ("'", '"', "`"):
            quote_char = char
            index += 1
            continue

        if char == "(":
            depth += 1

            if depth > MAX_AST_DEPTH:
                raise SQLGuardrailError(
                    "A consulta excede a profundidade máxima permitida."
                )

            index += 1
            continue

        if char == ")":
            depth = max(0, depth - 1)

        index += 1


# ============================================================
# COMPLEXIDADE DA AST
# ============================================================


def _validate_ast_complexity(query: exp.Expression) -> None:
    """Limita quantidade de nós, profundidade e número de CTEs."""

    node_count = 0
    pending_nodes = [(query, 1)]

    while pending_nodes:
        node, depth = pending_nodes.pop()

        node_count += 1

        if node_count > MAX_AST_NODES:
            raise SQLGuardrailError(
                "A consulta excede a quantidade máxima de nós na AST."
            )

        if depth > MAX_AST_DEPTH:
            raise SQLGuardrailError(
                "A consulta excede a profundidade máxima permitida."
            )

        for child in node.iter_expressions():
            pending_nodes.append((child, depth + 1))

    cte_count = sum(1 for _ in query.find_all(exp.CTE))

    if cte_count > MAX_CTES:
        raise SQLGuardrailError(
            "A consulta excede a quantidade máxima de CTEs permitida."
        )


# ============================================================
# VALIDAÇÃO DE COMANDOS
# ============================================================


def _validate_statement_nodes(query: exp.Expression) -> None:
    """Bloqueia escrita e comandos administrativos em toda a AST."""

    for node in query.walk():
        node_type = type(node).__name__

        if node_type in BLOCKED_STATEMENT_NODES:
            raise SQLGuardrailError(f"Operação SQL não permitida: {node_type.upper()}.")


# ============================================================
# VALIDAÇÃO DE FUNÇÕES
# ============================================================


def _get_function_names(node: exp.Func) -> set[str]:
    """
    Obtém o nome canônico de uma função SQL.

    Para funções anônimas, utiliza o nome informado no SQL.
    Para funções reconhecidas pelo SQLGlot, utiliza sql_name().

    Evita interpretar valores internos de expressões como
    nomes de funções.
    """

    if isinstance(node, exp.Anonymous):
        function_name = node.name

        if function_name:
            return {function_name.casefold()}

        return set()

    sql_name = node.sql_name()

    if sql_name:
        return {sql_name.casefold()}

    return set()


def _validate_functions(query: exp.Expression) -> None:
    """
    Aplica a política de funções permitidas.

    Funções explicitamente bloqueadas são rejeitadas primeiro.
    Qualquer função não autorizada também é rejeitada.

    Construções sintáticas seguras, como CASE, IF e operadores
    lógicos, não são tratadas como funções externas.
    """

    for node in query.walk():
        if not isinstance(node, exp.Func):
            continue

        node_type = type(node).__name__

        # Ignorar construções sintáticas que não representam
        # chamadas convencionais de funções.
        if node_type in SAFE_FUNCTION_EXPRESSIONS:
            continue

        function_names = _get_function_names(node)

        if not function_names:
            raise SQLGuardrailError(f"Função SQL não identificada: {node_type}.")

        blocked = function_names.intersection(BLOCKED_FUNCTIONS)

        if blocked:
            function_name = sorted(blocked)[0]

            raise SQLGuardrailError(f"Função bloqueada: {function_name}")

        allowed = function_names.intersection(ALLOWED_FUNCTIONS)

        if not allowed:
            function_name = sorted(function_names)[0]

            raise SQLGuardrailError(f"Função não permitida: {function_name}")


# ============================================================
# VALIDAÇÃO DE TABELAS
# ============================================================


def _validate_tables(
    query: exp.Expression,
    cte_names: set[str],
) -> None:
    """Permite somente tabelas autorizadas e CTEs declaradas."""

    for node in query.find_all(exp.Table):
        table_source = node.this

        if isinstance(table_source, exp.Func):
            function_names = _get_function_names(table_source)

            blocked = function_names.intersection(BLOCKED_FUNCTIONS)

            if blocked:
                function_name = sorted(blocked)[0]

                raise SQLGuardrailError(f"Função bloqueada: {function_name}")

            allowed = function_names.intersection(ALLOWED_FUNCTIONS)

            if not allowed:
                function_name = (
                    sorted(function_names)[0]
                    if function_names
                    else type(table_source).__name__
                )

                raise SQLGuardrailError(f"Função não permitida: {function_name}")

        table_name = node.name.casefold()

        if node.db or node.catalog:
            raise SQLGuardrailError(f"Tabela não autorizada: {node.sql()}")

        if table_name in cte_names:
            continue

        if not table_name:
            raise SQLGuardrailError("Fonte de tabela não autorizada.")

        if table_name not in ALLOWED_TABLES:
            raise SQLGuardrailError(f"Tabela não autorizada: {table_name}")


# ============================================================
# VALIDADOR PRINCIPAL
# ============================================================


def validate_query(sql_string: str) -> str:
    """
    Valida uma consulta SQL antes de permitir sua execução.

    Regras:
    1. Valida entrada e comprimento.
    2. Limita aninhamento antes do parsing.
    3. Aceita somente uma instrução SELECT como raiz.
    4. Bloqueia operações de escrita em qualquer nível.
    5. Bloqueia comandos administrativos.
    6. Limita tamanho e complexidade da AST.
    7. Aplica allowlist de funções SQL.
    8. Permite somente tabelas autorizadas ou CTEs declaradas.
    9. Bloqueia schemas e catálogos externos.

    Retorna:
        Query SQL normalizada.

    Lança:
        SQLGuardrailError: Quando uma regra de segurança é violada.
        SQLSyntaxError: Quando a consulta apresenta erro de sintaxe.
    """

    if not isinstance(sql_string, str) or not sql_string.strip():
        raise SQLGuardrailError("A query não pode estar vazia.")

    _validate_sql_length(sql_string)
    _validate_parenthesis_depth(sql_string)

    try:
        parsed_queries = sqlglot.parse(
            sql_string,
            read="duckdb",
        )

    except RecursionError as exc:
        raise SQLGuardrailError(
            "A consulta excede a profundidade máxima permitida."
        ) from exc

    except sqlglot.errors.ParseError as exc:
        raise SQLSyntaxError(
            sql_string,
            f"Sintaxe SQL invalida: {exc}",
        ) from exc

    parsed_queries = [query for query in parsed_queries if query is not None]

    if len(parsed_queries) != 1:
        raise SQLGuardrailError("Múltiplas instruções SQL não são permitidas.")

    query = parsed_queries[0]

    if not isinstance(query, exp.Select):
        raise SQLGuardrailError("Somente instruções SELECT são permitidas.")

    _validate_ast_complexity(query)
    _validate_statement_nodes(query)

    cte_names = {
        cte.alias_or_name.casefold()
        for cte in query.find_all(exp.CTE)
        if cte.alias_or_name
    }

    _validate_functions(query)
    _validate_tables(query, cte_names)

    return query.sql(dialect="duckdb")
