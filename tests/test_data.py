import duckdb
import pandas as pd
import pytest

from ai_analyst.data.db import build_database, connect_readonly
from ai_analyst.data.generate import generate_tables
from ai_analyst.data.schema import COLUMN_DOCS

N = 8_000


@pytest.fixture(scope="module")
def tables():
    return generate_tables(seed=7, n_orders=N)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("db") / "t.duckdb", seed=7, n_orders=N)


def test_same_seed_same_data():
    a = generate_tables(seed=1, n_orders=500)["orders"]
    b = generate_tables(seed=1, n_orders=500)["orders"]
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_different_data():
    a = generate_tables(seed=1, n_orders=500)["orders"]
    b = generate_tables(seed=2, n_orders=500)["orders"]
    assert not a["items_total"].equals(b["items_total"])


def test_referential_integrity(tables):
    o = tables["orders"]
    assert len(o) == N and o["order_id"].is_unique
    assert o["customer_id"].isin(tables["customers"]["customer_id"]).all()
    assert o["restaurant_id"].isin(tables["restaurants"]["restaurant_id"]).all()
    assert o["courier_id"].isin(tables["couriers"]["courier_id"]).all()


def test_city_consistency(tables):
    o, r = tables["orders"], tables["restaurants"].set_index("restaurant_id")
    c, k = tables["customers"].set_index("customer_id"), tables["couriers"].set_index("courier_id")
    assert (o["city"].to_numpy() == r.loc[o["restaurant_id"], "city"].to_numpy()).all()
    assert (o["city"].to_numpy() == c.loc[o["customer_id"], "city"].to_numpy()).all()
    assert (o["city"].to_numpy() == k.loc[o["courier_id"], "city"].to_numpy()).all()


def test_no_order_before_signup(tables):
    o, c = tables["orders"], tables["customers"].set_index("customer_id")
    signup = pd.to_datetime(c.loc[o["customer_id"], "signup_date"]).to_numpy()
    assert (o["order_ts"].to_numpy() > signup).all()


def test_canceled_orders_have_no_delivery_or_rating(tables):
    canceled = tables["orders"].query("status == 'canceled'")
    assert len(canceled) > 0
    assert canceled["delivery_minutes"].isna().all() and canceled["customer_rating"].isna().all()


def test_value_ranges(tables):
    o = tables["orders"]
    assert set(o["status"]) == {"delivered", "canceled"}
    assert o["items_total"].min() >= 10 and o["distance_km"].min() >= 0.4
    assert o["customer_rating"].dropna().between(1, 5).all()
    assert o["order_ts"].min() >= pd.Timestamp("2025-01-01") and o["order_ts"].max() < pd.Timestamp("2026-01-01")


def test_injected_pattern_lunch_peak_beats_dawn(tables):
    hours = tables["orders"]["order_ts"].dt.hour.value_counts()
    assert hours[12] > 5 * hours[4]


def test_injected_pattern_rain_slows_delivery_and_raises_cancellations(tables):
    o = tables["orders"]
    d = o.groupby("is_raining")["delivery_minutes"].mean()
    assert d[True] - d[False] > 2
    cancel = o.assign(c=o["status"].eq("canceled")).groupby("is_raining")["c"].mean()
    assert cancel[True] > cancel[False]


def test_db_row_counts(db_path):
    con = connect_readonly(db_path)
    assert con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == N
    con.close()


def test_schema_docs_match_database(db_path):
    con = connect_readonly(db_path)
    rows = con.execute("SELECT table_name, column_name FROM information_schema.columns "
                       "WHERE table_schema = 'main'").fetchall()
    con.close()
    actual = {(t, c) for t, c in rows}
    documented = {(t, c) for t, cols in COLUMN_DOCS.items() for c in cols}
    assert actual == documented


def test_connection_is_read_only(db_path):
    con = connect_readonly(db_path)
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM orders")
    with pytest.raises(duckdb.Error):
        con.execute("DROP TABLE orders")
    con.close()


def test_connection_blocks_file_access(db_path):
    con = connect_readonly(db_path)
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv('/etc/hosts')")
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")  # configuracao travada
    con.close()
