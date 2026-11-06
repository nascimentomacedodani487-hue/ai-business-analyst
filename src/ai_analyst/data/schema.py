"""Documentacao do schema. Na Etapa 3 vira contexto do prompt do LLM,
e um teste garante que ela nunca fica desatualizada em relacao ao banco."""

TABLE_DOCS = {
    "restaurants": "Restaurantes parceiros (um por linha).",
    "customers": "Clientes cadastrados.",
    "couriers": "Entregadores.",
    "orders": "Pedidos de 2025 (um por linha). Fato principal para analises.",
}

COLUMN_DOCS = {
    "restaurants": {
        "restaurant_id": "ID do restaurante (chave).",
        "name": "Nome do restaurante.",
        "category": "Pizza, Burger, Japanese, Brazilian, Healthy, Dessert ou Grocery.",
        "city": "Cidade: Sao Paulo, Rio de Janeiro, Belo Horizonte, Curitiba ou Recife (sem acento).",
        "avg_rating": "Nota media do restaurante (3.0 a 5.0).",
    },
    "customers": {
        "customer_id": "ID do cliente (chave).",
        "city": "Cidade do cliente.",
        "signup_date": "Data de cadastro (sempre antes de 2025).",
    },
    "couriers": {
        "courier_id": "ID do entregador (chave).",
        "city": "Cidade de atuacao.",
        "vehicle": "motorcycle, bicycle ou car.",
    },
    "orders": {
        "order_id": "ID do pedido (chave).",
        "customer_id": "FK para customers.",
        "restaurant_id": "FK para restaurants.",
        "courier_id": "FK para couriers.",
        "order_ts": "Data e hora do pedido (TIMESTAMP).",
        "city": "Cidade do pedido (= cidade do restaurante).",
        "status": "delivered (entregue) ou canceled (cancelado).",
        "items_total": "Valor dos itens em R$ (antes de taxa e desconto).",
        "delivery_fee": "Taxa de entrega em R$ (0 quando frete gratis).",
        "discount": "Desconto de cupom em R$ (0 quando sem cupom).",
        "payment_method": "credit_card, debit_card, pix, meal_voucher ou cash.",
        "distance_km": "Distancia restaurante-cliente em km.",
        "prep_minutes": "Minutos de preparo (preenchido tambem em cancelados).",
        "delivery_minutes": "Minutos de entrega apos sair do restaurante (NULL se cancelado).",
        "is_raining": "TRUE se choveu na cidade no dia do pedido.",
        "customer_rating": "Nota 1-5 dada pelo cliente (NULL se nao avaliou ou cancelado).",
    },
}


def describe_schema() -> str:
    """Texto do schema, pronto para ser usado como contexto de um prompt."""
    lines = []
    for table, doc in TABLE_DOCS.items():
        lines.append(f"TABLE {table} -- {doc}")
        for col, cdoc in COLUMN_DOCS[table].items():
            lines.append(f"  {col}: {cdoc}")
    return "\n".join(lines)
