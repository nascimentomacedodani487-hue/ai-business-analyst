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


class FakeStructuredLLMProvider:
    """Simula um provedor com suporte à geração JSON estruturada."""

    def __init__(self, response):
        self.response = response
        self.structured_calls = 0
        self.normal_calls = 0
        self.last_prompt = None
        self.last_system_prompt = None
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
        self.last_prompt = user_prompt
        self.last_system_prompt = system_prompt
        self.received_schema = response_schema
        self.received_tokens = max_output_tokens

        return self.response

    def generate_response(
        self,
        user_prompt,
        system_prompt,
        max_output_tokens=512,
    ):
        self.normal_calls += 1

        raise AssertionError("O planner deveria utilizar geração JSON estruturada.")


def make_response(objectives):
    return json.dumps(
        {"objectives": objectives},
        ensure_ascii=False,
    )


def make_behavior_payload(
    behavior_type,
    *,
    objectives=None,
    reason="Decisão fundamentada no schema disponível.",
    clarification_question=None,
    limitations=None,
):
    """Cria uma resposta comportamental estruturada."""

    return {
        "decision": {
            "type": behavior_type,
            "reason": reason,
            "clarification_question": clarification_question,
            "limitations": limitations or [],
        },
        "objectives": objectives if objectives is not None else [],
    }


def make_behavior_response(
    behavior_type,
    **kwargs,
):
    """Serializa uma resposta comportamental em JSON."""

    return json.dumps(
        make_behavior_payload(behavior_type, **kwargs),
        ensure_ascii=False,
    )


def make_objective(
    objective="Calcular o total de pedidos",
    question="Quantos pedidos foram realizados?",
):
    return {
        "objective": objective,
        "question": question,
    }


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

    provider = FakeStructuredLLMProvider(
        {
            "objectives": [
                {
                    "objective": "Calcular faturamento",
                    "question": "Qual foi o faturamento total?",
                }
            ]
        }
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan("Qual foi o faturamento total?")

    assert len(result) == 1
    assert result[0]["objective"] == "Calcular faturamento"

    assert provider.structured_calls == 1
    assert provider.normal_calls == 0

    assert provider.received_tokens == 2048
    assert provider.received_schema["type"] == "object"
    assert "objectives" in provider.received_schema["properties"]


# ==========================================================
# PLANEJAMENTO COM AVALIAÇÃO COMPORTAMENTAL
# ==========================================================


def test_plan_with_behavior_returns_single_objective():
    """Verifica answer com um único objetivo."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Quantos pedidos foram realizados?")

    assert len(result["objectives"]) == 1
    assert result["objectives"][0]["id"] == 1
    assert result["objectives"][0]["objective"] == ("Calcular o total de pedidos")

    assert result["behavior"]["type"] == "answer"
    assert result["behavior"]["reason"]
    assert result["behavior"]["clarification_question"] is None
    assert result["behavior"]["limitations"] == []
    assert result["behavior"]["supported_information"] == []
    assert result["behavior"]["sql_executed"] is False

    assert provider.calls == 1


def test_plan_with_behavior_returns_multiple_objectives():
    """Verifica answer com múltiplos objetivos independentes."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(
                    "Analisar pedidos mensais",
                    "Quantos pedidos foram realizados por mês?",
                ),
                make_objective(
                    "Analisar faturamento",
                    "Qual foi o faturamento mensal?",
                ),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior(
        "Analise a evolução dos pedidos e do faturamento."
    )

    assert len(result["objectives"]) == 2
    assert result["objectives"][0]["id"] == 1
    assert result["objectives"][1]["id"] == 2
    assert result["behavior"]["type"] == "answer"


def test_plan_with_behavior_returns_clarification_without_objectives():
    """Verifica ask_clarification sem iniciar consultas."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "ask_clarification",
            reason="O período da análise não foi especificado.",
            clarification_question="Qual período você deseja analisar?",
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Analise o desempenho dos pedidos.")

    assert result["objectives"] == []
    assert result["behavior"]["type"] == "ask_clarification"
    assert result["behavior"]["reason"]
    assert (
        result["behavior"]["clarification_question"]
        == "Qual período você deseja analisar?"
    )
    assert result["behavior"]["limitations"] == []
    assert result["behavior"]["sql_executed"] is False

    assert provider.calls == 1


def test_plan_with_behavior_returns_abstention_without_objectives():
    """Verifica abstain com limitações explícitas."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "abstain",
            reason="A solicitação depende de informações externas.",
            limitations=[
                "O schema não possui dados meteorológicos.",
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Qual era a temperatura durante cada entrega?")

    assert result["objectives"] == []
    assert result["behavior"]["type"] == "abstain"
    assert result["behavior"]["reason"]
    assert result["behavior"]["clarification_question"] is None
    assert result["behavior"]["limitations"] == [
        "O schema não possui dados meteorológicos.",
    ]
    assert result["behavior"]["sql_executed"] is False

    assert provider.calls == 1


def test_plan_with_behavior_preserves_supported_and_unsupported_objectives():
    """
    Verifica se o planejador preserva objetivos independentes
    quando há partes suportadas e não suportadas no schema.
    """

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(
                    "Analisar quantidade de pedidos",
                    "Quantos pedidos foram realizados?",
                ),
                make_objective(
                    "Analisar dados meteorológicos",
                    "Qual era a temperatura durante as entregas?",
                ),
            ],
            limitations=[
                "O schema não possui dados meteorológicos.",
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior(
        "Analise a quantidade de pedidos e a temperatura durante as entregas."
    )

    assert len(result["objectives"]) == 2
    assert result["behavior"]["type"] == "answer"
    assert len(result["behavior"]["limitations"]) == 1


def test_plan_with_behavior_accepts_maximum_five_objectives():
    """Verifica o limite máximo de objetivos no planejamento comportamental."""

    objectives = [
        make_objective(
            f"Objetivo {index}",
            f"Pergunta específica {index}?",
        )
        for index in range(1, MAX_OBJECTIVES + 1)
    ]

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=objectives,
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Faça uma análise completa.")

    assert len(result["objectives"]) == MAX_OBJECTIVES


@pytest.mark.parametrize(
    "question",
    ["", "   ", None],
)
def test_plan_with_behavior_rejects_empty_question(question):
    """Verifica que perguntas vazias não acionam o provedor."""

    provider = FakeLLMProvider("{}")
    planner = QueryPlanner(provider=provider)

    with pytest.raises(ValueError):
        planner.plan_with_behavior(question)

    assert provider.calls == 0


def test_plan_with_behavior_uses_database_schema():
    """Verifica se o schema é incluído no prompt comportamental."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    planner.plan_with_behavior("Quantos pedidos existem?")

    assert "TABLE orders" in provider.last_prompt
    assert "TABLE restaurants" in provider.last_prompt
    assert "TABLE customers" in provider.last_prompt
    assert "TABLE couriers" in provider.last_prompt


def test_plan_with_behavior_uses_behavior_system_prompt():
    """Verifica se o novo método utiliza seu próprio contexto."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    planner.plan_with_behavior("Quantos pedidos existem?")

    assert provider.last_system_prompt
    assert "planejador comportamental" in (provider.last_system_prompt.casefold())
    assert provider.last_max_output_tokens == 2048


def test_plan_with_behavior_uses_fallback_provider():
    """Verifica o fallback para generate_response."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Quantos pedidos foram realizados?")

    assert result["behavior"]["type"] == "answer"
    assert provider.calls == 1
    assert provider.last_max_output_tokens == 2048


def test_plan_with_behavior_uses_structured_json_generation():
    """Verifica o uso de generate_json_response."""

    provider = FakeStructuredLLMProvider(
        make_behavior_payload(
            "answer",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Quantos pedidos foram realizados?")

    assert result["behavior"]["type"] == "answer"
    assert len(result["objectives"]) == 1

    assert provider.structured_calls == 1
    assert provider.normal_calls == 0

    assert provider.received_tokens == 2048
    assert provider.received_schema["type"] == "object"
    assert "decision" in provider.received_schema["properties"]
    assert "objectives" in provider.received_schema["properties"]


def test_plan_with_behavior_rejects_invalid_json():
    """Verifica a rejeição de JSON inválido."""

    provider = FakeLLMProvider("Isso não é JSON.")
    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="JSON inválido"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_empty_response():
    """Verifica a rejeição de resposta vazia."""

    provider = FakeLLMProvider("")
    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="resposta vazia"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_accepts_markdown_json():
    """Verifica a compatibilidade com JSON cercado por Markdown."""

    payload = make_behavior_response(
        "answer",
        objectives=[
            make_objective(),
        ],
    )

    provider = FakeLLMProvider(f"```json\n{payload}\n```")

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Quantos pedidos existem?")

    assert result["behavior"]["type"] == "answer"
    assert len(result["objectives"]) == 1


def test_plan_with_behavior_rejects_non_object_payload():
    """Verifica a rejeição de respostas que não são objetos JSON."""

    provider = FakeLLMProvider(json.dumps(["resposta inválida"]))
    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="objeto JSON"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_unexpected_top_level_fields():
    """Verifica a rejeição de campos extras na raiz."""

    payload = make_behavior_payload(
        "answer",
        objectives=[
            make_objective(),
        ],
    )

    payload["sql"] = "SELECT 1"

    provider = FakeLLMProvider(json.dumps(payload, ensure_ascii=False))

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="somente"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_unexpected_decision_fields():
    """Verifica se o objeto decision aceita somente os campos previstos."""

    payload = make_behavior_payload(
        "answer",
        objectives=[
            make_objective(),
        ],
    )

    payload["decision"]["extra"] = "campo não permitido"

    provider = FakeLLMProvider(json.dumps(payload, ensure_ascii=False))

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="exatamente"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_unknown_behavior():
    """Verifica a rejeição de comportamentos não suportados."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "partial_answer",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="não suportado"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_empty_reason():
    """Verifica se uma justificativa é obrigatória."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(),
            ],
            reason="   ",
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="justificativa"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_answer_without_objectives():
    """Verifica que answer exige pelo menos um objetivo."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="entre 1 e 5"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_more_than_five_objectives():
    """Verifica o limite superior de objetivos."""

    objectives = [
        make_objective(
            f"Objetivo {index}",
            f"Pergunta específica {index}?",
        )
        for index in range(1, MAX_OBJECTIVES + 2)
    ]

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=objectives,
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="entre 1 e 5"):
        planner.plan_with_behavior("Faça uma análise completa.")


def test_plan_with_behavior_rejects_duplicate_objectives():
    """Verifica a reutilização da validação de duplicidade."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(
                    "Total de pedidos",
                    "Quantos pedidos existem?",
                ),
                make_objective(
                    "TOTAL DE PEDIDOS",
                    "QUANTOS PEDIDOS EXISTEM?",
                ),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="duplicado"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_invalid_objective_fields():
    """Verifica a validação dos campos dos objetivos."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                {
                    "objective": "Total de pedidos",
                    "question": "Quantos pedidos existem?",
                    "sql": "SELECT 1",
                },
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="somente"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_clarification_without_question():
    """Verifica que ask_clarification exige uma pergunta."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "ask_clarification",
            reason="O período não foi especificado.",
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="pergunta de esclarecimento"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_clarification_with_objectives():
    """Verifica que ask_clarification não pode planejar consultas."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "ask_clarification",
            clarification_question="Qual período deseja analisar?",
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="não pode apresentar objetivos"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_abstention_without_limitations():
    """Verifica que abstain exige pelo menos uma limitação."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "abstain",
            reason="A informação não está disponível.",
            limitations=[],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="pelo menos uma limitação"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_abstention_with_clarification():
    """Verifica que abstain não pode solicitar esclarecimentos."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "abstain",
            reason="A informação não existe no banco.",
            clarification_question="Qual período deseja analisar?",
            limitations=[
                "O schema não possui essa informação.",
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="não pode solicitar esclarecimentos"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_clarification_with_empty_objectives():
    """Verifica a coerência entre decisão e objetivos."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "ask_clarification",
            clarification_question="Qual período deseja analisar?",
            objectives=[],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Analise os pedidos.")

    assert result["objectives"] == []
    assert result["behavior"]["type"] == "ask_clarification"


@pytest.mark.parametrize(
    "limitations",
    [
        "Não é uma lista.",
        [""],
        ["   "],
        [1],
        [None],
    ],
)
def test_plan_with_behavior_rejects_invalid_limitations(limitations):
    """Verifica o tipo e o conteúdo das limitações."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "abstain",
            reason="A informação não está disponível.",
            limitations=limitations,
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_too_many_limitations():
    """Verifica o limite máximo de dez limitações."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "abstain",
            reason="A informação não está disponível.",
            limitations=[f"Limitação {index}" for index in range(11)],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="limite de dez"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_empty_clarification_question():
    """Verifica que uma pergunta de esclarecimento não pode ser vazia."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "ask_clarification",
            reason="Falta especificar o período.",
            clarification_question="   ",
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="não pode estar vazia"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_non_string_reason():
    """Verifica que reason deve ser texto."""

    payload = make_behavior_payload(
        "answer",
        objectives=[
            make_objective(),
        ],
    )

    payload["decision"]["reason"] = 123

    provider = FakeLLMProvider(json.dumps(payload, ensure_ascii=False))

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="justificativa"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_non_list_objectives():
    """Verifica que objectives precisa ser uma lista."""

    payload = make_behavior_payload(
        "answer",
    )

    payload["objectives"] = "objetivo inválido"

    provider = FakeLLMProvider(json.dumps(payload, ensure_ascii=False))

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="deve ser uma lista"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_rejects_abstention_with_objectives():
    """Verifica que abstain não pode produzir objetivos executáveis."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "abstain",
            reason="Os dados necessários não estão disponíveis.",
            limitations=[
                "O schema não contém dados meteorológicos.",
            ],
            objectives=[
                make_objective(),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    with pytest.raises(QueryPlanError, match="não pode apresentar objetivos"):
        planner.plan_with_behavior("Analise os pedidos.")


def test_plan_with_behavior_returns_normalized_ids():
    """Verifica que os objetivos recebem IDs sequenciais."""

    provider = FakeLLMProvider(
        make_behavior_response(
            "answer",
            objectives=[
                make_objective(
                    "Análise de faturamento",
                    "Qual foi o faturamento?",
                ),
                make_objective(
                    "Análise de pedidos",
                    "Quantos pedidos foram realizados?",
                ),
            ],
        )
    )

    planner = QueryPlanner(provider=provider)

    result = planner.plan_with_behavior("Analise os pedidos e faturamento.")

    assert [item["id"] for item in result["objectives"]] == [1, 2]
