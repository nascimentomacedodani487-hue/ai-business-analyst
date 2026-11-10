from ai_analyst.agent.business_analyst import BusinessAnalyst


def test_grouped_bar_chart_preserves_status_series_by_month():
    analyst = BusinessAnalyst()

    results = [
        {
            "order_month": "2025-01",
            "status": "canceled",
            "total_orders": 186,
        },
        {
            "order_month": "2025-01",
            "status": "delivered",
            "total_orders": 4901,
        },
        {
            "order_month": "2025-02",
            "status": "canceled",
            "total_orders": 172,
        },
        {
            "order_month": "2025-02",
            "status": "delivered",
            "total_orders": 4414,
        },
    ]

    chart = analyst._build_chart_config(
        results,
        question="Qual é a contagem de pedidos por status em cada mês?",
    )

    assert chart is not None
    assert chart["type"] == "bar"
    assert chart["label_key"] == "order_month"
    assert chart["value_key"] == "total_orders"

    assert chart["series"] == [
        {
            "data_key": "series_0",
            "label": "Canceled",
        },
        {
            "data_key": "series_1",
            "label": "Delivered",
        },
    ]

    assert chart["data"] == [
        {
            "order_month": "2025-01",
            "series_0": 186.0,
            "series_1": 4901.0,
        },
        {
            "order_month": "2025-02",
            "series_0": 172.0,
            "series_1": 4414.0,
        },
    ]


def test_monthly_revenue_remains_a_line_chart():
    analyst = BusinessAnalyst()

    results = [
        {
            "month": "2025-01",
            "total_revenue": 314492.87,
        },
        {
            "month": "2025-02",
            "total_revenue": 279689.32,
        },
        {
            "month": "2025-03",
            "total_revenue": 317836.60,
        },
    ]

    chart = analyst._build_chart_config(
        results,
        question="Qual foi o faturamento mensal?",
    )

    assert chart is not None
    assert chart["type"] == "line"
    assert chart["label_key"] == "month"
    assert chart["value_key"] == "total_revenue"
    assert "series" not in chart


def test_status_distribution_without_time_dimension_remains_pie():
    analyst = BusinessAnalyst()

    results = [
        {
            "status": "canceled",
            "total_orders": 100,
        },
        {
            "status": "delivered",
            "total_orders": 900,
        },
    ]

    chart = analyst._build_chart_config(
        results,
        question="Qual a distribuição de pedidos por status?",
    )

    assert chart is not None
    assert chart["type"] == "pie"
    assert chart["label_key"] == "status"
    assert chart["value_key"] == "total_orders"
    assert "series" not in chart


def test_wide_monthly_metrics_generate_grouped_bar_chart():
    analyst = BusinessAnalyst()

    results = [
        {
            "month": "2025-01",
            "delivered_orders": 4901,
            "canceled_orders": 186,
        },
        {
            "month": "2025-02",
            "delivered_orders": 4414,
            "canceled_orders": 172,
        },
    ]

    chart = analyst._build_chart_config(
        results,
        question=(
            "Mostre a quantidade de pedidos entregues e cancelados por mês em 2025."
        ),
    )

    assert chart is not None
    assert chart["type"] == "bar"
    assert chart["label_key"] == "month"

    assert chart["series"] == [
        {
            "data_key": "series_0",
            "label": "Delivered Orders",
        },
        {
            "data_key": "series_1",
            "label": "Canceled Orders",
        },
    ]

    assert chart["data"] == [
        {
            "month": "2025-01",
            "series_0": 4901.0,
            "series_1": 186.0,
        },
        {
            "month": "2025-02",
            "series_0": 4414.0,
            "series_1": 172.0,
        },
    ]
