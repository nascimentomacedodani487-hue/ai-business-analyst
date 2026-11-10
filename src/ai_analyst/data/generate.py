"""Gerador de dados SINTETICOS de um marketplace de delivery de comida.

Os dados sao ficticios e reproduziveis (seed fixa). Padroes injetados de
proposito, para que haja o que descobrir e para que os testes possam
verificar o gerador:
  - picos de pedidos no almoco e no jantar;
  - mais pedidos de sexta a domingo;
  - chuva aumenta o tempo de entrega e a taxa de cancelamento;
  - notas dos clientes caem quando o tempo total passa de ~45 min.
Qualquer "insight" encontrado nestes dados reflete essas regras, nao o mundo real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = ["Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Recife"]
CITY_WEIGHTS = np.array([0.38, 0.22, 0.15, 0.13, 0.12])
RAIN_PROB = {
    "Sao Paulo": 0.30,
    "Rio de Janeiro": 0.25,
    "Belo Horizonte": 0.22,
    "Curitiba": 0.28,
    "Recife": 0.35,
}
CATEGORIES = [
    "Pizza",
    "Burger",
    "Japanese",
    "Brazilian",
    "Healthy",
    "Dessert",
    "Grocery",
]
PREP_BASE_MIN = {
    "Pizza": 22,
    "Burger": 17,
    "Japanese": 25,
    "Brazilian": 20,
    "Healthy": 15,
    "Dessert": 12,
    "Grocery": 14,
}
BASKET_MULT = {
    "Pizza": 1.2,
    "Burger": 1.0,
    "Japanese": 1.5,
    "Brazilian": 1.1,
    "Healthy": 0.9,
    "Dessert": 0.6,
    "Grocery": 2.2,
}
VEHICLES = ["motorcycle", "bicycle", "car"]
VEHICLE_WEIGHTS = [0.75, 0.15, 0.10]
SPEED_KMH = {"motorcycle": 28.0, "bicycle": 14.0, "car": 22.0}
PAYMENTS = ["credit_card", "debit_card", "pix", "meal_voucher", "cash"]
PAYMENT_WEIGHTS = [0.38, 0.12, 0.35, 0.10, 0.05]
PEAK_HOURS = (12, 13, 19, 20)
HOUR_WEIGHTS = np.array(
    [
        1,
        1,
        0.5,
        0.3,
        0.2,
        0.2,
        0.5,
        1,
        2,
        2,
        3,
        8,
        10,
        8,
        4,
        3,
        3,
        5,
        9,
        12,
        11,
        8,
        4,
        2,
    ],
    dtype=float,
)
START_DATE, END_DATE = "2025-01-01", "2025-12-31"


def _restaurants(rng: np.random.Generator, n: int) -> pd.DataFrame:
    ids = np.arange(1, n + 1)
    return pd.DataFrame(
        {
            "restaurant_id": ids,
            "name": [f"Restaurante {i:03d}" for i in ids],
            "category": rng.choice(CATEGORIES, size=n),
            "city": rng.choice(CITIES, size=n, p=CITY_WEIGHTS),
            "avg_rating": np.round(np.clip(rng.normal(4.3, 0.35, n), 3.0, 5.0), 1),
        }
    )


def _customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    # Cadastros antes de 2025, para nunca existir pedido anterior ao cadastro.
    signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 1095, n), unit="D"
    )
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "city": rng.choice(CITIES, size=n, p=CITY_WEIGHTS),
            "signup_date": pd.Series(signup).dt.date,
        }
    )


def _couriers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "courier_id": np.arange(1, n + 1),
            "city": rng.choice(CITIES, size=n, p=CITY_WEIGHTS),
            "vehicle": rng.choice(VEHICLES, size=n, p=VEHICLE_WEIGHTS),
        }
    )


def _orders(rng, n, restaurants, customers, couriers) -> pd.DataFrame:
    days = pd.date_range(START_DATE, END_DATE, freq="D")
    day_w = np.where(days.day_of_week.to_numpy() >= 4, 1.25, 1.0)  # sex-dom: +25%
    day_idx = rng.choice(len(days), size=n, p=day_w / day_w.sum())
    hour = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    seconds = hour * 3600 + rng.integers(0, 3600, size=n)
    order_ts = days.to_numpy()[day_idx] + seconds.astype("timedelta64[s]")

    pop = rng.lognormal(
        0.0, 0.8, len(restaurants)
    )  # restaurantes populares vendem mais
    rest = restaurants.iloc[
        rng.choice(len(restaurants), size=n, p=pop / pop.sum())
    ].reset_index(drop=True)
    city, category = rest["city"].to_numpy(), rest["category"].to_numpy()

    # Cliente e entregador sempre da mesma cidade do restaurante.
    cust_id = np.zeros(n, dtype=np.int64)
    cour_id = np.zeros(n, dtype=np.int64)
    rain = np.zeros(n, dtype=bool)
    rain_by_city = {c: rng.random(len(days)) < RAIN_PROB[c] for c in CITIES}
    for c in CITIES:
        m = city == c
        k = int(m.sum())
        if k == 0:
            continue
        cust_id[m] = rng.choice(
            customers.loc[customers["city"] == c, "customer_id"].to_numpy(), size=k
        )
        cour_id[m] = rng.choice(
            couriers.loc[couriers["city"] == c, "courier_id"].to_numpy(), size=k
        )
        rain[m] = rain_by_city[c][day_idx[m]]

    vehicle = couriers.set_index("courier_id").loc[cour_id, "vehicle"].to_numpy()
    speed = pd.Series(vehicle).map(SPEED_KMH).to_numpy()
    distance = np.round(np.clip(rng.gamma(2.2, 1.6, n) + 0.4, 0.4, 15.0), 1)
    is_peak = np.isin(hour, PEAK_HOURS)

    prep = np.clip(
        rng.normal(
            pd.Series(category).map(PREP_BASE_MIN).to_numpy() + 4 * is_peak, 4.0
        ),
        5,
        60,
    )
    delivery = (
        distance / speed * 60 * (1 + 0.25 * rain)
        + 3 * is_peak
        + 2 * rain
        + rng.normal(0, 3, n)
    )
    delivery = np.clip(delivery, 3, None)

    p_cancel = np.clip(0.03 + 0.02 * rain + 0.0015 * np.maximum(prep - 30, 0), 0, 0.5)
    canceled = rng.random(n) < p_cancel

    items = np.round(
        np.clip(
            rng.lognormal(np.log(45), 0.5, n)
            * pd.Series(category).map(BASKET_MULT).to_numpy(),
            10,
            600,
        ),
        2,
    )
    fee = np.where(rng.random(n) < 0.15, 0.0, np.round(3.99 + 1.1 * distance, 2))
    discount = np.where(
        rng.random(n) < 0.22,
        np.round(items * rng.choice([0.10, 0.15, 0.20, 0.30], size=n), 2),
        0.0,
    )

    rating_raw = np.clip(
        np.round(
            4.7 - 0.03 * np.maximum(prep + delivery - 45, 0) + rng.normal(0, 0.6, n)
        ),
        1,
        5,
    )
    rated = (rng.random(n) < 0.6) & ~canceled

    df = (
        pd.DataFrame(
            {
                "customer_id": cust_id,
                "restaurant_id": rest["restaurant_id"].to_numpy(),
                "courier_id": cour_id,
                "order_ts": order_ts,
                "city": city,
                "status": np.where(canceled, "canceled", "delivered"),
                "items_total": items,
                "delivery_fee": fee,
                "discount": discount,
                "payment_method": rng.choice(PAYMENTS, size=n, p=PAYMENT_WEIGHTS),
                "distance_km": distance,
                "prep_minutes": np.round(prep, 1),
                "delivery_minutes": pd.Series(np.round(delivery, 1)).where(~canceled),
                "is_raining": rain,
                "customer_rating": pd.Series(
                    rating_raw.astype(int), dtype="Int64"
                ).where(rated),
            }
        )
        .sort_values("order_ts", kind="stable")
        .reset_index(drop=True)
    )
    df.insert(0, "order_id", np.arange(1, n + 1))
    return df


def generate_tables(seed: int = 42, n_orders: int = 60_000) -> dict[str, pd.DataFrame]:
    """Gera as 4 tabelas. Mesma seed => mesmos dados."""
    rng = np.random.default_rng(seed)
    restaurants = _restaurants(rng, 200)
    customers = _customers(rng, 5_000)
    couriers = _couriers(rng, 300)
    orders = _orders(rng, n_orders, restaurants, customers, couriers)
    return {
        "restaurants": restaurants,
        "customers": customers,
        "couriers": couriers,
        "orders": orders,
    }
