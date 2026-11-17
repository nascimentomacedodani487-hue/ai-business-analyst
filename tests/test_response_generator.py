import json

import pytest

from ai_analyst.agent.response_generator import (
    MAX_SYNTHESIS_COLUMNS,
    MAX_SYNTHESIS_PAYLOAD_CHARS,
    MAX_SYNTHESIS_ROWS,
    ResponseGenerator,
)
from ai_analyst.llm.base import LLMResponseError


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


# ==========================================================
# TESTES LEGADOS DO GERADOR
# ==========================================================


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
                make_query(status="pending"),
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
        {
            "month": f"month_{index:02d}",
            "count": index,
        }
        for index in range(1, 25)
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


# ==========================================================
# FÁBRICAS DOS TESTES DE EVIDÊNCIAS
# ==========================================================


def make_structured_response(answer, claims):
    """Cria uma resposta JSON simulada do provedor."""

    return json.dumps(
        {
            "answer": answer,
            "claims": claims,
        },
        ensure_ascii=False,
    )


def make_evidence(
    query_id=1,
    row_index=0,
    column="total",
    value=100,
):
    """Cria uma referência de evidência."""

    return {
        "query_id": query_id,
        "row_index": row_index,
        "column": column,
        "value": value,
    }


def make_claim(
    text="Foram encontrados 100 pedidos.",
    evidence=None,
):
    """Cria uma afirmação estruturada."""

    return {
        "claim": text,
        "evidence": ([make_evidence()] if evidence is None else evidence),
    }


# ==========================================================
# TESTES DE GERAÇÃO ESTRUTURADA COM EVIDÊNCIAS
# ==========================================================


def test_generate_with_evidence_returns_validated_claims():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(),
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    result = generator.generate_with_evidence(
        "Quantos pedidos existem?",
        "SELECT COUNT(*) AS total FROM orders",
        [{"total": 100}],
    )

    assert result["answer"] == "Foram encontrados 100 pedidos."

    assert result["answer_evidence"] == [
        make_claim(),
    ]

    assert provider.calls == 1
    assert provider.last_max_output_tokens == 1200

    assert "claims" in provider.last_prompt
    assert "query_id" in provider.last_prompt
    assert "row_index" in provider.last_prompt


def test_generate_multi_with_evidence_supports_query_identifiers():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(
                            query_id=7,
                            column="total_pedidos",
                            value=100,
                        )
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            query_id=7,
            results=[
                {
                    "total_pedidos": 100,
                }
            ],
        )
    ]

    result = generator.generate_multi_with_evidence(
        "Quantos pedidos existem?",
        queries,
    )

    assert result["answer"] == "Foram encontrados 100 pedidos."

    assert result["answer_evidence"][0]["evidence"][0] == {
        "query_id": 7,
        "row_index": 0,
        "column": "total_pedidos",
        "value": 100,
    }

    assert provider.calls == 1


def test_generate_with_evidence_rejects_invented_value():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 101 pedidos.",
            [
                make_claim(
                    text="Foram encontrados 101 pedidos.",
                    evidence=[
                        make_evidence(value=101),
                    ],
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="não corresponde",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )

    assert provider.calls == 1


def test_generate_with_evidence_rejects_unknown_query():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(query_id=999),
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="consulta referenciada.*não existe",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )


def test_generate_multi_with_evidence_rejects_failed_query():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(query_id=2),
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            query_id=1,
            results=[{"total": 100}],
        ),
        make_query(
            query_id=2,
            status="failure",
            results=[],
            error="Erro SQL",
        ),
    ]

    with pytest.raises(
        LLMResponseError,
        match="não foi concluída",
    ):
        generator.generate_multi_with_evidence(
            "Analise os pedidos.",
            queries,
        )


def test_generate_with_evidence_rejects_missing_row():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(row_index=5),
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="linha referenciada.*não existe",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )


def test_generate_with_evidence_rejects_missing_column():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(column="nonexistent_column"),
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="coluna referenciada.*não existe",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )


def test_generate_with_evidence_rejects_truncated_cell():
    provider = FakeLLMProvider(
        make_structured_response(
            "A descrição contém informações.",
            [
                make_claim(
                    text="A descrição contém informações.",
                    evidence=[
                        make_evidence(
                            column="description",
                            value="A" * 300,
                        )
                    ],
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="célula referenciada.*foi truncada",
    ):
        generator.generate_with_evidence(
            "Analise as descrições.",
            "SELECT description FROM orders",
            [{"description": "A" * 301}],
        )


def test_generate_with_evidence_rejects_claim_without_evidence():
    provider = FakeLLMProvider(
        make_structured_response(
            "Os pedidos aumentaram.",
            [
                make_claim(
                    evidence=[],
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="não possui evidências",
    ):
        generator.generate_with_evidence(
            "Analise os pedidos.",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )


def test_generate_with_evidence_rejects_invalid_json():
    provider = FakeLLMProvider("Esta não é uma resposta JSON válida.")

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="JSON inválido",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )


def test_generate_with_evidence_accepts_markdown_json_fence():
    response = make_structured_response(
        "Foram encontrados 100 pedidos.",
        [
            make_claim(),
        ],
    )

    provider = FakeLLMProvider(f"```json\n{response}\n```")

    generator = ResponseGenerator(provider=provider)

    result = generator.generate_with_evidence(
        "Quantos pedidos existem?",
        "SELECT COUNT(*) AS total FROM orders",
        [{"total": 100}],
    )

    assert result["answer"] == "Foram encontrados 100 pedidos."
    assert len(result["answer_evidence"]) == 1


def test_generate_with_evidence_rejects_missing_structured_fields():
    provider = FakeLLMProvider(
        json.dumps(
            {
                "answer": "Análise concluída.",
            }
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="campos obrigatórios",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )


def test_generate_with_evidence_does_not_call_provider_without_results():
    provider = FakeLLMProvider()

    generator = ResponseGenerator(provider=provider)

    result = generator.generate_with_evidence(
        "Quantos pedidos existem?",
        "SELECT COUNT(*) AS total FROM orders",
        [],
    )

    assert "Não foram encontrados dados" in result["answer"]
    assert result["answer_evidence"] == []

    assert provider.calls == 0


def test_generate_multi_with_evidence_does_not_call_provider_without_data():
    provider = FakeLLMProvider()

    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            query_id=1,
            results=[],
        ),
        make_query(
            query_id=2,
            status="failure",
            error="Erro SQL",
        ),
    ]

    result = generator.generate_multi_with_evidence(
        "Analise os pedidos.",
        queries,
    )

    assert "Não foi possível" in result["answer"]
    assert result["answer_evidence"] == []

    assert provider.calls == 0


def test_generate_multi_with_evidence_preserves_input():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(
                            query_id=1,
                        )
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            query_id=1,
            results=[{"total": 100}],
        )
    ]

    original = json.loads(json.dumps(queries))

    generator.generate_multi_with_evidence(
        "Analise os pedidos.",
        queries,
    )

    assert queries == original


# ==========================================================
# TESTES DE INTEGRIDADE DOS METADADOS DE TRUNCAMENTO
# ==========================================================


def test_generate_with_evidence_informs_row_truncation():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados registros de pedidos.",
            [],
        )
    )

    generator = ResponseGenerator(provider=provider)

    results = [{"order_id": index} for index in range(MAX_SYNTHESIS_ROWS + 10)]

    generator.generate_with_evidence(
        "Analise os pedidos.",
        "SELECT order_id FROM orders",
        results,
    )

    assert '"results_truncated":true' in provider.last_prompt
    assert '"rows_truncated":true' in provider.last_prompt
    assert '"provided_results":30' in provider.last_prompt

    assert len(results) == MAX_SYNTHESIS_ROWS + 10


def test_generate_with_evidence_informs_column_truncation():
    provider = FakeLLMProvider(
        make_structured_response(
            "Os dados apresentados foram analisados.",
            [],
        )
    )

    generator = ResponseGenerator(provider=provider)

    results = [{f"column_{index}": index for index in range(MAX_SYNTHESIS_COLUMNS + 5)}]

    generator.generate_with_evidence(
        "Analise os dados.",
        "SELECT * FROM orders",
        results,
    )

    assert '"columns_truncated":true' in provider.last_prompt
    assert '"omitted_columns":5' in provider.last_prompt

    assert "column_19" in provider.last_prompt
    assert "column_20" not in provider.last_prompt


def test_generate_with_evidence_informs_cell_truncation():
    provider = FakeLLMProvider(
        make_structured_response(
            "A amostra foi analisada.",
            [],
        )
    )

    generator = ResponseGenerator(provider=provider)

    generator.generate_with_evidence(
        "Analise a descrição.",
        "SELECT description FROM orders",
        [
            {
                "description": "A" * 500,
            }
        ],
    )

    assert '"cells_truncated":true' in provider.last_prompt
    assert '"truncated_cells"' in provider.last_prompt


def test_serialize_payload_respects_strict_character_limit():
    generator_payload = ResponseGenerator._build_multi_payload(
        [
            make_query(
                query_id=1,
                results=[
                    {
                        f"column_{column}": "A" * 500
                        for column in range(MAX_SYNTHESIS_COLUMNS)
                    }
                    for _ in range(MAX_SYNTHESIS_ROWS)
                ],
                total_results=100,
            )
        ]
    )

    original_count = len(generator_payload[0]["results"])

    serialized = ResponseGenerator._serialize_payload(
        generator_payload,
    )

    parsed = json.loads(serialized)

    assert len(serialized) <= MAX_SYNTHESIS_PAYLOAD_CHARS

    assert len(parsed[0]["results"]) < original_count

    assert parsed[0]["results_truncated"] is True

    # A serialização não deve modificar o payload original.
    assert len(generator_payload[0]["results"]) == original_count


# ==========================================================
# TESTES DE VALIDAÇÃO DE ENTRADAS E IDENTIFICADORES
# ==========================================================


@pytest.mark.parametrize(
    "total_results",
    [
        True,
        -1,
        0,
    ],
)
def test_generate_multi_with_evidence_rejects_invalid_total_results(
    total_results,
):
    provider = FakeLLMProvider()

    generator = ResponseGenerator(provider=provider)

    query = make_query(
        results=[{"total": 100}],
        total_results=total_results,
    )

    with pytest.raises(
        ValueError,
        match="total_results inválido",
    ):
        generator.generate_multi_with_evidence(
            "Analise os pedidos.",
            [query],
        )

    assert provider.calls == 0


def test_generate_multi_with_evidence_rejects_duplicate_query_ids():
    provider = FakeLLMProvider()

    generator = ResponseGenerator(provider=provider)

    queries = [
        make_query(
            query_id=1,
            results=[{"total": 100}],
        ),
        make_query(
            query_id=1,
            results=[{"total": 200}],
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Identificador duplicado",
    ):
        generator.generate_multi_with_evidence(
            "Analise os pedidos.",
            queries,
        )

    assert provider.calls == 0


def test_generate_multi_with_evidence_rejects_boolean_query_id():
    provider = FakeLLMProvider()

    generator = ResponseGenerator(provider=provider)

    query = make_query(
        query_id=True,
        results=[{"total": 100}],
    )

    with pytest.raises(
        ValueError,
        match="identificador.*texto ou inteiro",
    ):
        generator.generate_multi_with_evidence(
            "Analise os pedidos.",
            [query],
        )

    assert provider.calls == 0


def test_generate_with_evidence_rejects_boolean_row_index():
    provider = FakeLLMProvider(
        make_structured_response(
            "Foram encontrados 100 pedidos.",
            [
                make_claim(
                    evidence=[
                        make_evidence(
                            row_index=True,
                        )
                    ]
                )
            ],
        )
    )

    generator = ResponseGenerator(provider=provider)

    with pytest.raises(
        LLMResponseError,
        match="índice da linha.*inválido",
    ):
        generator.generate_with_evidence(
            "Quantos pedidos existem?",
            "SELECT COUNT(*) AS total FROM orders",
            [{"total": 100}],
        )
