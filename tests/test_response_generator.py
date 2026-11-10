import pytest

from ai_analyst.agent.response_generator import (
    MAX_SYNTHESIS_PAYLOAD_CHARS,
    MAX_SYNTHESIS_ROWS,
    ResponseGenerator,
)


class FakeLLMProvider:
    """Simula respostas de um provedor de IA."""

    def __init__(self, response="Análise concluída."):
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


def make_query(
    query_id=1,
    status="success",
    results=None,
    **kwargs,
):
    query = {
        "id": query_id,
        "objective": f"Objetivo {query_id}",
        "question": f"Pergunta {query_id}",
        "status": status,
        "results": results or [],
        "total_results": len(results or []),
    }
    query.update(kwargs)
    return query


def test_generate_preserves_existing_behavior():
    provider = FakeLLMProvider("Foram encontrados 10 pedidos.")
    generator = ResponseGenerator(provider=provider)

    answer = generator.generate(
        "Quantos pedidos existem?",
        "SELECT COUNT(*) AS total FROM orders",
        [{"total": 10}],
    )

    assert answer == "Foram encontrados 10 pedidos."
    assert provider.calls == 1
    assert "Quantos pedidos existem?" in provider.last_prompt
    assert '"total":10' in provider.last_prompt
    assert provider.last_max_output_tokens == 512


def test_generate_returns_message_without_results():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    answer = generator.generate(
        "Quantos pedidos existem?",
        "SELECT 1",
        [],
    )

    assert "Não foram encontrados dados" in answer
    assert provider.calls == 0


def test_generate_multi_with_successful_queries():
    provider = FakeLLMProvider("Os pedidos cresceram.")
    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            1,
            results=[{"total_pedidos": 100}],
        ),
        make_query(
            2,
            results=[{"faturamento": 5000}],
        ),
    ]

    answer = generator.generate_multi(
        "Analise pedidos e faturamento.",
        queries,
    )

    assert answer == "Os pedidos cresceram."
    assert provider.calls == 1
    assert "total_pedidos" in provider.last_prompt
    assert "faturamento" in provider.last_prompt
    assert provider.last_max_output_tokens == 900


def test_generate_multi_identifies_partial_failures():
    provider = FakeLLMProvider("Uma análise foi concluída.")
    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            1,
            results=[{"total": 100}],
        ),
        make_query(
            2,
            status="failure",
            error="Binder Error",
        ),
    ]

    generator.generate_multi(
        "Analise os pedidos.",
        queries,
    )

    assert '"status":"success"' in provider.last_prompt
    assert '"status":"failure"' in provider.last_prompt
    assert "Binder Error" in provider.last_prompt


def test_generate_multi_does_not_call_provider_without_data():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(1, results=[]),
        make_query(
            2,
            status="failure",
            error="Erro SQL",
        ),
    ]

    answer = generator.generate_multi(
        "Analise os pedidos.",
        queries,
    )

    assert "Não foi possível" in answer
    assert provider.calls == 0


def test_generate_multi_rejects_empty_question():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    with pytest.raises(ValueError, match="pergunta original"):
        generator.generate_multi(
            "",
            [make_query()],
        )

    assert provider.calls == 0


def test_generate_multi_rejects_empty_queries():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    with pytest.raises(ValueError, match="pelo menos uma consulta"):
        generator.generate_multi(
            "Analise os pedidos.",
            [],
        )

    assert provider.calls == 0


def test_generate_multi_rejects_invalid_status():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    with pytest.raises(ValueError, match="Status inválido"):
        generator.generate_multi(
            "Analise os pedidos.",
            [
                make_query(
                    status="pending",
                )
            ],
        )

    assert provider.calls == 0


def test_generate_multi_rejects_invalid_results():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    query = make_query()
    query["results"] = "resultado inválido"

    with pytest.raises(ValueError, match="devem ser uma lista"):
        generator.generate_multi(
            "Analise os pedidos.",
            [query],
        )


def test_generate_multi_rejects_non_dictionary_rows():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    query = make_query()
    query["results"] = ["resultado inválido"]

    with pytest.raises(ValueError, match="devem ser objetos"):
        generator.generate_multi(
            "Analise os pedidos.",
            [query],
        )


def test_generate_multi_limits_rows_without_mutating_original():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    results = [{"order_id": index} for index in range(MAX_SYNTHESIS_ROWS + 10)]

    queries = [
        make_query(
            results=results,
            total_results=len(results),
        )
    ]

    generator.generate_multi(
        "Analise os pedidos.",
        queries,
    )

    assert len(results) == MAX_SYNTHESIS_ROWS + 10
    assert len(queries[0]["results"]) == MAX_SYNTHESIS_ROWS + 10

    assert '"results_truncated":true' in provider.last_prompt


def test_generate_multi_limits_columns():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    result = {f"column_{index}": index for index in range(30)}

    generator.generate_multi(
        "Analise os pedidos.",
        [
            make_query(
                results=[result],
                total_results=1,
            )
        ],
    )

    assert "column_19" in provider.last_prompt
    assert "column_20" not in provider.last_prompt


def test_generate_multi_limits_cell_length():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    generator.generate_multi(
        "Analise os pedidos.",
        [
            make_query(
                results=[{"description": "A" * 500}],
                total_results=1,
            )
        ],
    )

    assert "A" * 301 not in provider.last_prompt
    assert "A" * 300 in provider.last_prompt


def test_generate_multi_limits_total_payload():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    results = [{"description": "A" * 300} for _ in range(MAX_SYNTHESIS_ROWS)]

    generator.generate_multi(
        "Analise os pedidos.",
        [
            make_query(
                results=results,
                total_results=100,
            )
        ],
    )

    assert len(provider.last_prompt) < (MAX_SYNTHESIS_PAYLOAD_CHARS + 5000)

    assert '"results_truncated":true' in provider.last_prompt


def test_generate_multi_does_not_mutate_input():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            results=[{"total": 100}],
        )
    ]

    original = {
        key: value.copy() if isinstance(value, list) else value
        for key, value in queries[0].items()
    }

    generator.generate_multi(
        "Analise os pedidos.",
        queries,
    )

    assert queries[0] == original


def test_generate_multi_accepts_only_success_or_failure():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    for status in ["pending", "partial", "running", "unknown"]:
        with pytest.raises(ValueError, match="Status inválido"):
            generator.generate_multi(
                "Analise os pedidos.",
                [
                    make_query(
                        status=status,
                        results=[],
                    )
                ],
            )


def test_response_generator_supports_gemini_alias():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(gemini=provider)

    assert generator.provider is provider
    assert generator.gemini is provider


def test_response_generator_rejects_conflicting_providers():
    provider_a = FakeLLMProvider()
    provider_b = FakeLLMProvider()

    with pytest.raises(ValueError, match="apenas um"):
        ResponseGenerator(
            provider=provider_a,
            gemini=provider_b,
        )


def test_generate_multi_includes_all_24_rows_when_payload_fits():
    provider = FakeLLMProvider()
    generator = ResponseGenerator(provider=provider)

    results = [
        {"month": f"month_{index:02d}", "count": index} for index in range(1, 25)
    ]

    generator.generate_multi(
        "Analise os pedidos por mes.",
        [
            make_query(
                results=results,
                total_results=24,
            )
        ],
    )

    assert '"month":"month_24"' in provider.last_prompt
    assert '"provided_results":24' in provider.last_prompt
    assert '"omitted_results":0' in provider.last_prompt
    assert '"results_truncated":false' in provider.last_prompt
