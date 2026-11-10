"""
Validação determinística da capacidade do schema para responder
a objetivos analíticos específicos.

Não utiliza chamadas de LLM.
"""

import re
import unicodedata

from sqlglot import exp, parse_one

from ai_analyst.data.schema import COLUMN_DOCS, TABLE_DOCS


class UnsupportedAnalysisError(ValueError):
    """A análise solicitada não pode ser realizada com o schema atual."""


class SQLIntentMismatchError(ValueError):
    """A SQL gerada não está alinhada à dimensão solicitada."""


PRODUCT_INTENT_PATTERNS = (
    r"\bpor\s+produtos?\b",
    r"\bprodutos?\s+(?:com|mais|menos)\b",
    r"\btop\s+\d+\s+produtos?\b",
    r"\b(?:faturamento|receita|vendas?)\s+(?:por|de|dos?|das?)\s+produtos?\b",
    r"\bsku(?:s)?\b",
    r"\bproducts?\s+(?:by|with|most|top)\b",
    r"\btop\s+\d+\s+products?\b",
)

PRODUCT_IDENTIFIER_COLUMNS = {
    "product_id",
    "product_name",
    "product",
    "sku",
    "item_id",
    "item_name",
    "produto_id",
    "nome_produto",
    "codigo_produto",
    "id_produto",
    "produto",
}

ORDER_REFERENCE_COLUMNS = {
    "order_id",
    "orderid",
    "pedido_id",
    "id_pedido",
}

LINE_AMOUNT_COLUMNS = {
    "line_total",
    "item_total",
    "items_total",
    "subtotal",
    "amount",
    "valor_item",
    "total_item",
    "valor_total_item",
    "sales_amount",
}

UNIT_PRICE_COLUMNS = {
    "unit_price",
    "price",
    "preco_unitario",
    "valor_unitario",
}

QUANTITY_COLUMNS = {
    "quantity",
    "qty",
    "quantidade",
}


def _normalize_text(value: str) -> str:
    """Normaliza texto, removendo acentos e diferenças de caixa."""

    normalized = unicodedata.normalize("NFKD", str(value))

    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).lower()


def _normalize_identifier(value: str) -> str:
    """Normaliza identificadores de tabelas e colunas."""

    return re.sub(
        r"[^a-z0-9_]",
        "",
        _normalize_text(value),
    )


def _requests_product_level_analysis(
    objective: str,
    question: str,
) -> bool:
    """
    Identifica objetivos que exigem dados individualizados por produto.

    A identificação da intenção não bloqueia a análise sozinha:
    a decisão final também depende da estrutura real do schema.
    """

    text = _normalize_text(f"{objective or ''}\n{question or ''}")

    return any(re.search(pattern, text) for pattern in PRODUCT_INTENT_PATTERNS)


def _is_product_identifier(column_name: str) -> bool:
    """Verifica se uma coluna identifica um produto ou item individual."""

    name = _normalize_identifier(column_name)

    if name in PRODUCT_IDENTIFIER_COLUMNS:
        return True

    return any(
        token in name
        for token in (
            "product_id",
            "product_name",
            "produto_id",
            "nome_produto",
            "codigo_produto",
            "item_id",
            "item_name",
            "sku",
        )
    )


def _schema_has_product_granularity(
    column_docs: dict,
) -> bool:
    """
    Verifica se existe uma estrutura capaz de relacionar:

    - Um pedido.
    - Um produto individual.
    - Um valor por item ou quantidade e preço unitário.

    Um total agregado por pedido não é considerado granularidade
    suficiente para calcular faturamento por produto.
    """

    for columns in column_docs.values():
        normalized_columns = {_normalize_identifier(column) for column in columns}

        has_order_reference = bool(normalized_columns & ORDER_REFERENCE_COLUMNS)

        has_product_identifier = any(
            _is_product_identifier(column) for column in normalized_columns
        )

        has_line_amount = bool(normalized_columns & LINE_AMOUNT_COLUMNS)

        has_quantity_and_price = bool(normalized_columns & QUANTITY_COLUMNS) and bool(
            normalized_columns & UNIT_PRICE_COLUMNS
        )

        if (
            has_order_reference
            and has_product_identifier
            and (has_line_amount or has_quantity_and_price)
        ):
            return True

    return False


def _unsupported_product_message(
    column_docs: dict,
) -> str:
    """Explica por que a análise por produto não está disponível."""

    order_columns = {
        _normalize_identifier(column) for column in column_docs.get("orders", {})
    }

    if "items_total" in order_columns:
        return (
            "Análise em nível de produto indisponível. "
            "O schema não possui identificação individual dos produtos "
            "associados a cada pedido. A coluna `orders.items_total` "
            "representa o valor agregado dos itens do pedido, sem "
            "discriminar produtos individuais, quantidades ou valores "
            "por produto. Portanto, não é possível calcular o "
            "faturamento por produto com os dados atuais. "
            "Nenhuma consulta substituta será executada."
        )

    return (
        "Análise em nível de produto indisponível. "
        "O schema não possui uma estrutura que relacione pedidos, "
        "produtos individuais e valores por item ou quantidade e "
        "preço unitário. Não é possível calcular o faturamento por "
        "produto com os dados atuais."
    )


def validate_analysis_capability(
    objective: str,
    question: str,
    *,
    table_docs: dict | None = None,
    column_docs: dict | None = None,
) -> None:
    """
    Impede a geração de SQL quando a análise solicitada depende
    de uma granularidade inexistente no banco.
    """

    if not _requests_product_level_analysis(objective, question):
        return

    effective_columns = COLUMN_DOCS if column_docs is None else column_docs

    if not _schema_has_product_granularity(effective_columns):
        raise UnsupportedAnalysisError(_unsupported_product_message(effective_columns))


def validate_sql_intent_alignment(
    objective: str,
    question: str,
    sql: str,
    *,
    table_docs: dict | None = None,
    column_docs: dict | None = None,
) -> None:
    """
    Verifica se a SQL de uma análise por produto realmente utiliza
    uma identificação de produto na projeção do SELECT.

    Exemplo de divergência bloqueada:
    objetivo = faturamento por produto
    SQL = faturamento agrupado por restaurante.
    """

    if not _requests_product_level_analysis(objective, question):
        return

    effective_tables = TABLE_DOCS if table_docs is None else table_docs

    effective_columns = COLUMN_DOCS if column_docs is None else column_docs

    if not _schema_has_product_granularity(effective_columns):
        raise UnsupportedAnalysisError(_unsupported_product_message(effective_columns))

    product_table_names = {
        _normalize_identifier(table)
        for table in effective_tables
        if any(
            marker in _normalize_identifier(table)
            for marker in ("product", "produto", "item", "sku")
        )
    }

    product_column_names = {
        _normalize_identifier(column)
        for columns in effective_columns.values()
        for column in columns
        if _is_product_identifier(column)
    }

    try:
        tree = parse_one(sql, read="duckdb")
    except Exception as exc:
        raise SQLIntentMismatchError(
            "Não foi possível verificar o alinhamento semântico da SQL "
            "gerada com o objetivo solicitado."
        ) from exc

    product_aliases = set()

    for table in tree.find_all(exp.Table):
        table_name = _normalize_identifier(table.name)

        if table_name in product_table_names:
            product_aliases.add(_normalize_identifier(table.alias_or_name))

    select = tree.find(exp.Select)

    if select is None:
        raise SQLIntentMismatchError(
            "A SQL gerada não contém uma projeção SELECT válida."
        )

    selected_columns = []

    for projection in select.expressions:
        if isinstance(projection, exp.Column):
            selected_columns.append(projection)
        else:
            selected_columns.extend(projection.find_all(exp.Column))

    selects_product = any(
        (
            _normalize_identifier(column.table) in product_aliases
            if column.table
            else False
        )
        or (_normalize_identifier(column.name) in product_column_names)
        for column in selected_columns
    )

    if not selects_product:
        raise SQLIntentMismatchError(
            "A SQL gerada não está alinhada ao objetivo de analisar "
            "produtos: ela não apresenta uma identificação de produto "
            "no resultado. A consulta não será executada para evitar "
            "substituir a análise solicitada por outra dimensão."
        )
