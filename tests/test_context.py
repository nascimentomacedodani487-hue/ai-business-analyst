from ai_analyst.agent.context import build_system_prompt


def test_prompt_is_generated():
    prompt = build_system_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 500


def test_prompt_contains_all_tables():
    prompt = build_system_prompt()

    for table in ["orders", "restaurants", "customers", "couriers"]:
        assert table in prompt


def test_prompt_contains_security_rules():
    prompt = build_system_prompt()

    assert "SELECT" in prompt
    assert "arquivos externos" in prompt
    assert "bancos de dados" in prompt


def test_prompt_contains_schema_columns():
    prompt = build_system_prompt()

    for column in ["order_id", "customer_id", "restaurant_id", "items_total"]:
        assert column in prompt
