import pytest

from ai_analyst.agent.capability_validator import (
    SQLIntentMismatchError,
    UnsupportedAnalysisError,
    validate_analysis_capability,
    validate_sql_intent_alignment,
)

FUTURE_TABLES = {
    "orders": "Pedidos.",
    "products": "Catálogo de produtos.",
    "order_items": "Itens individuais de cada pedido.",
    "restaurants": "Restaurantes.",
}

FUTURE_COLUMNS = {
    "orders": {
        "order_id",
        "restaurant_id",
    },
    "products": {
        "product_id",
        "name",
    },
    "order_items": {
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
    },
    "restaurants": {
        "restaurant_id",
        "name",
    },
}


def test_product_analysis_is_rejected_by_current_schema():
    """O schema atual não permite análise de faturamento por produto."""

    with pytest.raises(
        UnsupportedAnalysisError,
        match="orders.items_total",
    ):
        validate_analysis_capability(
            objective="Identificar os top 10 produtos com maior faturamento",
            question="Quais são os 10 produtos com maior faturamento?",
        )


def test_restaurant_analysis_is_not_blocked():
    """Análise por restaurante não deve ser bloqueada."""

    validate_analysis_capability(
        objective="Analisar faturamento por restaurante",
        question=(
            "Compare o faturamento por restaurante "
            "considerando o valor total dos itens dos pedidos."
        ),
    )


def test_product_analysis_is_allowed_with_item_level_schema():
    """Um schema futuro com itens individualizados deve permitir a análise."""

    validate_analysis_capability(
        objective="Identificar os top 10 produtos com maior faturamento",
        question="Quais são os 10 produtos com maior faturamento?",
        table_docs=FUTURE_TABLES,
        column_docs=FUTURE_COLUMNS,
    )


def test_product_analysis_rejects_sql_about_restaurants():
    """Uma SQL de restaurantes não pode responder sobre produtos."""

    misaligned_sql = """
        SELECT
            r.name AS restaurant_name,
            SUM(o.items_total) AS total_revenue
        FROM orders AS o
        JOIN restaurants AS r
            ON r.restaurant_id = o.restaurant_id
        GROUP BY r.name
        ORDER BY total_revenue DESC
        LIMIT 10
    """

    with pytest.raises(
        SQLIntentMismatchError,
        match="não está alinhada",
    ):
        validate_sql_intent_alignment(
            objective="Identificar os top 10 produtos com maior faturamento",
            question="Quais são os 10 produtos com maior faturamento?",
            sql=misaligned_sql,
            table_docs=FUTURE_TABLES,
            column_docs=FUTURE_COLUMNS,
        )


def test_product_analysis_accepts_aligned_sql():
    """Uma SQL que seleciona produtos e agrega seus itens é aceita."""

    aligned_sql = """
        SELECT
            p.name AS product_name,
            SUM(oi.quantity * oi.unit_price) AS total_revenue
        FROM order_items AS oi
        JOIN products AS p
            ON p.product_id = oi.product_id
        GROUP BY p.name
        ORDER BY total_revenue DESC
        LIMIT 10
    """

    validate_sql_intent_alignment(
        objective="Identificar os top 10 produtos com maior faturamento",
        question="Quais são os 10 produtos com maior faturamento?",
        sql=aligned_sql,
        table_docs=FUTURE_TABLES,
        column_docs=FUTURE_COLUMNS,
    )
