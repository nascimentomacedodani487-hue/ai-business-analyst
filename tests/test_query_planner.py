import json

import pytest

from ai_analyst.agent.query_planner import (
    MAX_OBJECTIVES,
    QueryPlanError,
    QueryPlanner,
)


class FakeLLMProvider:
    """Simula respostas de um provedor de IA."""

    def __init__(self, response):
        self.response = response
        self.last_prompt = None
        self.last_system_prompt = None
        self.last_max_output_tokens = None
        self.calls = 0

    def generate_response(
        self,
        user_prompt,
        system_prompt,
        max_output_tokens=512,
    ):
        self.calls += 1
        self.last_prompt = user_prompt
        self.last_system_prompt = system_prompt
        self.last_max_output_tokens = max_output_tokens

        return self.response


def make_response(objectives):
    return json.dumps(
        {"objectives": objectives},
        ensure_ascii=False,
    )


def test_plan_single_objective():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Calcular o total de pedidos",
                    "question": "Quantos pedidos foram realizados?",
                }
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan("Quantos pedidos foram realizados?")

    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["objective"] == "Calcular o total de pedidos"
    assert provider.calls == 1


def test_plan_multiple_objectives():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Analisar pedidos mensais",
                    "question": "Quantos pedidos foram realizados por mês?",
                },
                {
                    "objective": "Analisar faturamento",
                    "question": "Qual foi o faturamento mensal?",
                },
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan("Analise a evolução dos pedidos e do faturamento.")

    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2


def test_plan_accepts_maximum_five_objectives():
    objectives = [
        {
            "objective": f"Objetivo {i}",
            "question": f"Pergunta específica {i}",
        }
        for i in range(1, MAX_OBJECTIVES + 1)
    ]

    provider = FakeLLMProvider(make_response(objectives))
    planner = QueryPlanner(provider=provider)

    result = planner.plan("Faça uma análise completa.")

    assert len(result) == 5


def test_plan_rejects_more_than_five_objectives():
    objectives = [
        {
            "objective": f"Objetivo {i}",
            "question": f"Pergunta específica {i}",
        }
        for i in range(1, 7)
    ]

    provider = FakeLLMProvider(make_response(objectives))
    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="entre 1 e 5"):
        planner.plan("Faça uma análise completa.")


@pytest.mark.parametrize(
    "question",
    ["", "   ", None],
)
def test_plan_rejects_empty_question(question):
    provider = FakeLLMProvider("{}")
    planner = QueryPlanner(provider=provider)

    with pytest.raises(ValueError):
        planner.plan(question)

    assert provider.calls == 0


def test_plan_rejects_invalid_json():
    provider = FakeLLMProvider("Isso não é JSON.")
    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="JSON inválido"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_empty_response():
    provider = FakeLLMProvider("")
    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="resposta vazia"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_invalid_payload():
    provider = FakeLLMProvider(json.dumps(["objetivo inválido"]))

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="objeto JSON"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_unexpected_payload_keys():
    provider = FakeLLMProvider(
        json.dumps(
            {
                "objectives": [],
                "sql": "SELECT * FROM orders",
            }
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="somente a chave"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_empty_objectives():
    provider = FakeLLMProvider(make_response([]))

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="entre 1 e 5"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_empty_objective_fields():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "",
                    "question": "Quantos pedidos existem?",
                }
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="campos vazios"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_duplicate_objectives():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Total de pedidos",
                    "question": "Quantos pedidos existem?",
                },
                {
                    "objective": "TOTAL DE PEDIDOS",
                    "question": "QUANTOS PEDIDOS EXISTEM?",
                },
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="duplicado"):
        planner.plan("Analise os pedidos.")


def test_plan_rejects_unexpected_objective_fields():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Total de pedidos",
                    "question": "Quantos pedidos existem?",
                    "sql": "SELECT 1",
                }
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="somente"):
        planner.plan("Analise os pedidos.")


def test_plan_uses_database_schema():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Total de pedidos",
                    "question": "Quantos pedidos existem?",
                }
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    planner.plan("Quantos pedidos existem?")

    assert "TABLE orders" in provider.last_prompt
    assert "TABLE restaurants" in provider.last_prompt
    assert "TABLE customers" in provider.last_prompt
    assert "TABLE couriers" in provider.last_prompt


def test_plan_uses_expected_provider_configuration():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Total de pedidos",
                    "question": "Quantos pedidos existem?",
                }
            ]
        )
    )

    planner = QueryPlanner(provider=provider)

    planner.plan("Quantos pedidos existem?")

    assert provider.last_system_prompt
    assert provider.last_max_output_tokens == 2048


def test_plan_supports_gemini_compatibility_alias():
    provider = FakeLLMProvider(
        make_response(
            [
                {
                    "objective": "Total de pedidos",
                    "question": "Quantos pedidos existem?",
                }
            ]
        )
    )

    planner = QueryPlanner(gemini=provider)

    assert planner.provider is provider
    assert planner.gemini is provider


def test_plan_rejects_conflicting_providers():
    provider_a = FakeLLMProvider("{}")
    provider_b = FakeLLMProvider("{}")

    with pytest.raises(ValueError, match="apenas um"):
        QueryPlanner(
            provider=provider_a,
            gemini=provider_b,
        )


def test_plan_uses_structured_json_generation():
    """Verifica se o planner prioriza a geração JSON estruturada."""

    class StructuredLLMProvider:
        def __init__(self):
            self.structured_calls = 0
            self.normal_calls = 0
            self.received_schema = None
            self.received_tokens = None

        def generate_json_response(
            self,
            user_prompt,
            system_prompt,
            response_schema,
            max_output_tokens=2048,
        ):
            self.structured_calls += 1
            self.received_schema = response_schema
            self.received_tokens = max_output_tokens

            return json.dumps(
                {
                    "objectives": [
                        {
                            "objective": "Calcular faturamento",
                            "question": "Qual foi o faturamento total?",
                        }
                    ]
                },
                ensure_ascii=False,
            )

        def generate_response(
            self,
            user_prompt,
            system_prompt,
            max_output_tokens=512,
        ):
            self.normal_calls += 1
            raise AssertionError("O planner deveria utilizar geração JSON estruturada.")

    provider = StructuredLLMProvider()
    planner = QueryPlanner(provider=provider)

    result = planner.plan("Qual foi o faturamento total?")

    assert len(result) == 1
    assert result[0]["objective"] == "Calcular faturamento"

    assert provider.structured_calls == 1
    assert provider.normal_calls == 0

    assert provider.received_tokens == 2048
    assert provider.received_schema["type"] == "object"
    assert "objectives" in provider.received_schema["properties"]
