import pytest

from ai_analyst.data.history import AnalysisHistory


@pytest.fixture
def history(tmp_path):
    database_path = tmp_path / "summary_history.sqlite3"
    return AnalysisHistory(database_path)


def test_summary_excludes_full_analysis_payload(history):
    """A listagem resumida não deve devolver dados completos da análise."""

    history.save_analysis(
        analysis_id="summary-001",
        question="Analise os pedidos.",
        status="success",
        llm_metrics={"tokens": {"total": 500}},
        steps=[{"name": "sql_generation", "duration_ms": 100}],
        sql="SELECT * FROM orders",
        queries=[
            {
                "id": 1,
                "objective": "Pedidos por status",
                "question": "Quantos pedidos existem por status?",
                "status": "success",
                "sql": "SELECT status, COUNT(*) FROM orders GROUP BY status",
                "results": [{"status": "delivered", "total": 100}],
                "total_results": 1,
                "failure_type": None,
            }
        ],
        results=[{"total": 100}],
        total_results=1,
        answer="Foram encontrados 100 pedidos.",
        chart={"type": "bar"},
        provider="gemini",
        model="gemini-2.5-flash",
    )

    response = history.list_analysis_summaries()

    assert response["pagination"]["total"] == 1

    item = response["items"][0]

    assert item["analysis_id"] == "summary-001"
    assert item["question"] == "Analise os pedidos."
    assert item["status"] == "success"
    assert item["provider"] == "gemini"
    assert item["model"] == "gemini-2.5-flash"
    assert item["partial_success"] is False

    forbidden_fields = {
        "sql",
        "results",
        "queries",
        "llm_metrics",
        "steps",
        "answer",
        "chart",
    }

    assert forbidden_fields.isdisjoint(item.keys())


def test_summary_identifies_partial_success(history):
    """A listagem deve identificar análises com sucesso parcial."""

    history.save_analysis(
        analysis_id="summary-partial-001",
        question="Analise pedidos e faturamento.",
        status="success",
        queries=[
            {
                "id": 1,
                "status": "success",
                "results": [{"total": 100}],
            },
            {
                "id": 2,
                "status": "failure",
                "failure_type": "sql_execution_error",
                "results": [],
            },
        ],
    )

    item = history.list_analysis_summaries()["items"][0]

    assert item["partial_success"] is True


def test_summary_identifies_unsupported_capability(history):
    """Consultas não suportadas também caracterizam sucesso parcial."""

    history.save_analysis(
        analysis_id="summary-unsupported-001",
        question="Analise pedidos e produtos.",
        status="success",
        queries=[
            {
                "id": 1,
                "status": "success",
                "results": [{"total": 100}],
            },
            {
                "id": 2,
                "status": "failure",
                "failure_type": "unsupported_capability",
                "results": [],
            },
        ],
    )

    item = history.list_analysis_summaries()["items"][0]

    assert item["partial_success"] is True


def test_summary_does_not_mark_all_successful_queries_as_partial(history):
    history.save_analysis(
        analysis_id="summary-success-001",
        question="Analise os pedidos.",
        status="success",
        queries=[
            {"id": 1, "status": "success"},
            {"id": 2, "status": "success"},
        ],
    )

    item = history.list_analysis_summaries()["items"][0]

    assert item["partial_success"] is False


def test_summary_returns_safe_error_message(history):
    """A listagem não deve expor mensagens nem tipos internos de exceção."""

    internal_message = (
        "RuntimeError: caminho interno C:\\private\\database "
        "SELECT * FROM internal_table"
    )

    history.save_analysis(
        analysis_id="summary-failure-001",
        question="Analise os pedidos.",
        status="failure",
        error_type="LLMProviderError",
        error_message=internal_message,
    )

    item = history.list_analysis_summaries()["items"][0]

    assert item["error_type"] == "Falha no serviço de IA"
    assert item["error_message"] == ("O serviço de IA apresentou uma falha.")

    assert internal_message not in str(item)
    assert "LLMProviderError" not in str(item)
    assert "internal_table" not in str(item)
    assert "C:\\private\\database" not in str(item)

    # O registro original continua preservado para diagnóstico interno.
    stored = history.get_analysis("summary-failure-001")

    assert stored is not None
    assert stored["error_type"] == "LLMProviderError"
    assert stored["error_message"] == internal_message


def test_summary_uses_generic_message_for_unknown_error_type(history):
    """Tipos desconhecidos não devem revelar mensagens internas."""

    internal_message = "Detalhes privados de infraestrutura."

    history.save_analysis(
        analysis_id="summary-unknown-error-001",
        question="Analise os pedidos.",
        status="failure",
        error_type="InternalDatabaseException",
        error_message=internal_message,
    )

    item = history.list_analysis_summaries()["items"][0]

    assert item["error_type"] == "Falha interna"
    assert item["error_message"] == (
        "A análise não pôde ser concluída. "
        "Os detalhes técnicos não são exibidos nesta listagem."
    )

    assert internal_message not in str(item)
    assert "InternalDatabaseException" not in str(item)

    # Tipos desconhecidos também permanecem disponíveis internamente.
    stored = history.get_analysis("summary-unknown-error-001")

    assert stored is not None
    assert stored["error_type"] == "InternalDatabaseException"
    assert stored["error_message"] == internal_message


def test_summary_preserves_complete_details_in_separate_method(history):
    """
    A representação resumida não deve alterar a recuperação completa.
    """

    internal_message = "Falha técnica simulada."

    history.save_analysis(
        analysis_id="summary-details-001",
        question="Analise os pedidos.",
        status="failure",
        error_type="RuntimeError",
        error_message=internal_message,
        sql="SELECT COUNT(*) FROM orders",
        queries=[
            {
                "id": 1,
                "status": "success",
                "sql": "SELECT COUNT(*) FROM orders",
                "results": [{"total": 100}],
            }
        ],
        results=[{"total": 100}],
        total_results=1,
        answer="Resposta de teste.",
    )

    summary = history.list_analysis_summaries()["items"][0]
    details = history.get_analysis("summary-details-001")

    assert summary["error_message"] != internal_message

    assert summary["error_type"] == "Falha interna"

    assert details is not None
    assert details["error_type"] == "RuntimeError"
    assert details["error_message"] == internal_message
    assert details["sql"] == "SELECT COUNT(*) FROM orders"
    assert details["results"] == [{"total": 100}]
    assert details["answer"] == "Resposta de teste."
    assert len(details["queries"]) == 1


def test_list_analyses_preserves_legacy_full_response(history):
    """O método antigo continua compatível com consumidores existentes."""

    history.save_analysis(
        analysis_id="legacy-list-001",
        question="Analise os pedidos.",
        status="success",
        sql="SELECT COUNT(*) FROM orders",
        results=[{"total": 100}],
    )

    item = history.list_analyses()["items"][0]

    assert item["sql"] == "SELECT COUNT(*) FROM orders"
    assert item["results"] == [{"total": 100}]


def test_summary_preserves_filters_and_pagination(history):
    for index in range(3):
        history.save_analysis(
            analysis_id=f"summary-page-{index}",
            question=f"Pergunta {index}",
            status="success" if index < 2 else "failure",
        )

    response = history.list_analysis_summaries(
        limit=1,
        offset=0,
        status="success",
    )

    assert len(response["items"]) == 1
    assert response["pagination"]["total"] == 2
    assert response["pagination"]["limit"] == 1
    assert response["pagination"]["offset"] == 0
    assert response["pagination"]["has_more"] is True
