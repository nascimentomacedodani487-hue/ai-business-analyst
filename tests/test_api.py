from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from ai_analyst.agent.business_analyst import BusinessAnalyst
from ai_analyst.api.main import analysis_cache, app
from ai_analyst.data.history import AnalysisHistory
from ai_analyst.data.schema import TABLE_DOCS
from ai_analyst.llm.base import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

client = TestClient(app)


def create_test_history(monkeypatch, tmp_path):
    """Substitui o histórico global por um banco SQLite temporário."""

    history = AnalysisHistory(tmp_path / "test_analysis_history.sqlite3")

    monkeypatch.setattr(
        "ai_analyst.api.main.analysis_history",
        history,
    )

    return history


def seed_test_history(history):
    """Insere registros de exemplo no histórico."""

    history.save_analysis(
        analysis_id="analysis-success-001",
        question="Quantos pedidos foram entregues?",
        status="success",
        total_time_ms=1250.5,
        cache_hit=False,
        llm_metrics={
            "analysis_id": "analysis-success-001",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "requests": {
                "total": 2,
                "successful": 2,
                "failed": 0,
            },
            "tokens": {
                "input": 100,
                "output": 50,
                "total": 150,
            },
        },
        steps=[
            {
                "name": "sql_generation",
                "status": "completed",
                "duration_ms": 500,
            },
            {
                "name": "response_generation",
                "status": "completed",
                "duration_ms": 300,
            },
        ],
    )

    history.save_analysis(
        analysis_id="analysis-failure-001",
        question="Qual é a previsão de vendas?",
        status="failure",
        total_time_ms=800.0,
        cache_hit=False,
        llm_metrics={
            "analysis_id": "analysis-failure-001",
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "requests": {
                "total": 1,
                "successful": 0,
                "failed": 1,
            },
        },
        steps=[],
        error_type="LLMProviderError",
        error_message="Provedor indisponível.",
    )


# ============================================================
# TESTES GERAIS DA API
# ============================================================


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_request_returns_valid_id():
    """Verifica se cada requisição recebe um identificador válido."""

    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    request_id = response.headers["X-Request-ID"]

    UUID(request_id)


def test_requests_receive_unique_ids():
    """Verifica se requisições diferentes recebem IDs diferentes."""

    first_response = client.get("/health")
    second_response = client.get("/health")

    first_id = first_response.headers["X-Request-ID"]
    second_id = second_response.headers["X-Request-ID"]

    assert first_id != second_id


def test_analyze_rejects_short_question():
    response = client.post(
        "/analyze",
        json={"question": "Oi"},
    )

    assert response.status_code == 422


# ============================================================
# TESTES DE TRATAMENTO DE ERROS DA ANÁLISE
# ============================================================


def test_analyze_handles_rate_limit_error(monkeypatch, tmp_path):
    create_test_history(monkeypatch, tmp_path)
    """Verifica o HTTP 503 e a identificação da análise."""

    def simulate_quota_error(self, question):
        raise LLMRateLimitError("Limite de requisições atingido.")

    monkeypatch.setattr(
        BusinessAnalyst,
        "analyze",
        simulate_quota_error,
    )

    response = client.post(
        "/analyze",
        json={"question": "Quantos pedidos tivemos?"},
    )

    assert response.status_code == 503

    detail = response.json()["detail"]

    assert isinstance(detail, dict)

    assert "limite de requisições" in (detail["message"].lower())

    UUID(detail["analysis_id"])


def test_analyze_handles_timeout_error(monkeypatch, tmp_path):
    create_test_history(monkeypatch, tmp_path)
    """Verifica o HTTP 504 e a identificação da análise."""

    def simulate_timeout(self, question):
        raise LLMTimeoutError("Tempo limite excedido.")

    monkeypatch.setattr(
        BusinessAnalyst,
        "analyze",
        simulate_timeout,
    )

    response = client.post(
        "/analyze",
        json={"question": "Quantos pedidos tivemos?"},
    )

    assert response.status_code == 504

    detail = response.json()["detail"]

    assert isinstance(detail, dict)

    assert "demorou demais" in detail["message"].lower()

    UUID(detail["analysis_id"])


def test_analyze_handles_provider_error(monkeypatch, tmp_path):
    create_test_history(monkeypatch, tmp_path)
    """Verifica o HTTP 502 e a identificação da análise."""

    def simulate_provider_error(self, question):
        raise LLMProviderError("Provedor indisponível.")

    monkeypatch.setattr(
        BusinessAnalyst,
        "analyze",
        simulate_provider_error,
    )

    response = client.post(
        "/analyze",
        json={"question": "Quantos pedidos tivemos?"},
    )

    assert response.status_code == 502

    detail = response.json()["detail"]

    assert isinstance(detail, dict)

    assert "serviço de inteligência artificial" in (detail["message"].lower())

    UUID(detail["analysis_id"])


# ============================================================
# TESTES DE CACHE E EXECUÇÃO DAS ANÁLISES
# ============================================================


def test_api_reuses_cached_analysis(monkeypatch, tmp_path):
    """Verifica o cache e a diferença entre as duas Timelines."""

    analysis_cache.clear()
    history = create_test_history(monkeypatch, tmp_path)

    calls = {
        "sql": 0,
        "execution": 0,
        "response": 0,
    }

    class FakeGeminiClient:
        pass

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.create_llm_provider",
        FakeGeminiClient,
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        lambda: "test-v1",
    )

    def fake_generate(self, question):
        calls["sql"] += 1
        return "SELECT COUNT(*) AS total FROM orders"

    def fake_execute(self, sql):
        calls["execution"] += 1
        return [{"total": 100}]

    def fake_response(self, question, sql, results):
        calls["response"] += 1
        return "Foram encontrados 100 pedidos."

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.SQLGenerator.generate",
        fake_generate,
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.SQLExecutor.execute",
        fake_execute,
    )

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.ResponseGenerator.generate",
        fake_response,
    )

    question = "Quantos pedidos existem no teste de cache da API?"

    first_response = client.post(
        "/analyze",
        json={"question": question},
    )

    second_response = client.post(
        "/analyze",
        json={"question": question},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_result = first_response.json()
    second_result = second_response.json()

    assert first_result["cache_hit"] is False
    assert second_result["cache_hit"] is True

    first_steps = first_result.pop("steps")
    second_steps = second_result.pop("steps")

    first_result.pop("cache_hit")
    second_result.pop("cache_hit")

    first_metrics = first_result.pop("metrics")
    second_metrics = second_result.pop("metrics")

    first_analysis_id = first_result.pop("analysis_id")
    second_analysis_id = second_result.pop("analysis_id")

    assert first_analysis_id != second_analysis_id

    first_llm_metrics = first_result.pop("llm_metrics")
    second_llm_metrics = second_result.pop("llm_metrics")

    assert first_llm_metrics["analysis_id"] == first_analysis_id
    assert second_llm_metrics["analysis_id"] == second_analysis_id

    assert first_result == second_result

    first_step_names = [step["name"] for step in first_steps]

    assert "sql_generation" in first_step_names
    assert "response_generation" in first_step_names

    assert all(step["status"] == "completed" for step in first_steps)

    second_step_names = [step["name"] for step in second_steps]

    assert "cache_lookup" in second_step_names
    assert "cache_hit" in second_step_names

    assert "sql_generation" not in second_step_names
    assert "response_generation" not in second_step_names

    assert all(step["status"] == "completed" for step in second_steps)

    assert first_metrics["cache_hit"] is False
    assert "sql_generation_ms" in first_metrics
    assert "sql_execution_ms" in first_metrics
    assert "response_generation_ms" in first_metrics

    assert second_metrics["cache_hit"] is True
    assert "total_time_ms" in second_metrics

    assert calls["sql"] == 1
    assert calls["execution"] == 1
    assert calls["response"] == 1

    # As duas análises devem ter sido persistidas.
    assert history.count_analyses() == 2

    first_saved = history.get_analysis(first_analysis_id)
    second_saved = history.get_analysis(second_analysis_id)

    assert first_saved is not None
    assert second_saved is not None

    assert first_saved["status"] == "success"
    assert second_saved["status"] == "success"

    assert first_saved["cache_hit"] is False
    assert second_saved["cache_hit"] is True

    analysis_cache.clear()


# ============================================================
# TESTES DO CATÁLOGO DE DADOS
# ============================================================


def test_data_catalog_returns_documented_tables(monkeypatch):
    """Verifica a estrutura e o conteúdo do catálogo de dados."""

    class FakeConnection:
        def __init__(self):
            self.current_table = None
            self.last_query = None
            self.description = None
            self.closed = False

        def execute(self, query):
            self.last_query = query

            if query.startswith("DESCRIBE "):
                self.current_table = query.split()[-1]
                self.description = None
                return self

            if query.startswith("SELECT COUNT(*)"):
                return self

            if query.startswith("SELECT *"):
                self.description = [
                    ("id",),
                    ("status",),
                ]
                return self

            raise AssertionError(f"Consulta inesperada: {query}")

        def fetchall(self):
            if self.last_query.startswith("DESCRIBE "):
                return [
                    ("id", "INTEGER", "NO", None, None, None),
                    ("status", "VARCHAR", "YES", None, None, None),
                ]

            if self.last_query.startswith("SELECT *"):
                return [
                    (1, "delivered"),
                ]

            raise AssertionError(f"fetchall inesperado para: {self.last_query}")

        def fetchone(self):
            return (42,)

        def close(self):
            self.closed = True

    connection = FakeConnection()

    monkeypatch.setattr(
        "ai_analyst.api.main.connect_readonly",
        lambda: connection,
    )

    response = client.get("/data/catalog")

    assert response.status_code == 200

    data = response.json()

    assert data["database"] == "DuckDB"
    assert data["dataset"] == "Synthetic Delivery Dataset"
    assert data["period"] == "2025"

    assert data["tables_count"] == len(TABLE_DOCS)
    assert len(data["tables"]) == len(TABLE_DOCS)

    returned_tables = {table["name"]: table for table in data["tables"]}

    assert set(returned_tables) == set(TABLE_DOCS)

    for table_name in TABLE_DOCS:
        table = returned_tables[table_name]

        assert table["description"] == TABLE_DOCS[table_name]
        assert table["total_rows"] == 42

        assert len(table["columns"]) == 2

        assert table["columns"][0]["name"] == "id"
        assert table["columns"][0]["type"] == "INTEGER"
        assert table["columns"][0]["nullable"] is False

        assert table["columns"][1]["name"] == "status"
        assert table["columns"][1]["type"] == "VARCHAR"
        assert table["columns"][1]["nullable"] is True

        assert len(table["sample"]) == 1

        assert table["sample"][0] == {
            "id": 1,
            "status": "delivered",
        }

    assert connection.closed is True


def test_data_catalog_handles_database_error(monkeypatch):
    """Verifica o tratamento de erros e o fechamento da conexão."""

    class FakeConnection:
        def __init__(self):
            self.closed = False

        def execute(self, query):
            raise RuntimeError("Database unavailable")

        def close(self):
            self.closed = True

    connection = FakeConnection()

    monkeypatch.setattr(
        "ai_analyst.api.main.connect_readonly",
        lambda: connection,
    )

    response = client.get("/data/catalog")

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "Não foi possível consultar o catálogo de dados."
    )

    assert connection.closed is True


# ============================================================
# TESTES DE MÉTRICAS DE IA
# ============================================================


def test_llm_metrics_endpoint():
    from ai_analyst.llm.metrics import llm_metrics

    llm_metrics.reset()

    try:
        llm_metrics.record_success(
            model="gemini-2.5-flash",
            duration_ms=120.0,
            prompt_tokens=100,
            output_tokens=50,
            thinking_tokens=10,
            total_tokens=160,
        )

        llm_metrics.record_retry()

        response = client.get("/metrics/llm")

        assert response.status_code == 200

        data = response.json()

        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-2.5-flash"
        assert data["pricing_tier"] == "free"

        assert data["requests"]["total"] == 1
        assert data["requests"]["successful"] == 1
        assert data["requests"]["retries"] == 1

        assert data["tokens"]["input"] == 100
        assert data["tokens"]["output"] == 50
        assert data["tokens"]["thinking"] == 10
        assert data["tokens"]["total"] == 160

        assert data["cost_estimate_usd"] == 0.0

    finally:
        llm_metrics.reset()


# ============================================================
# TESTES DO HISTÓRICO DE ANÁLISES
# ============================================================


def test_analysis_history_lists_records(monkeypatch, tmp_path):
    """Verifica se a API lista análises persistidas."""

    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    response = client.get("/analyses/history")

    assert response.status_code == 200

    data = response.json()

    assert len(data["items"]) == 2

    assert data["pagination"]["total"] == 2
    assert data["pagination"]["limit"] == 20
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["has_more"] is False

    returned_ids = {item["analysis_id"] for item in data["items"]}

    assert returned_ids == {
        "analysis-success-001",
        "analysis-failure-001",
    }


def test_analysis_history_supports_pagination(monkeypatch, tmp_path):
    """Verifica os parâmetros limit e offset."""

    history = create_test_history(monkeypatch, tmp_path)

    for index in range(3):
        history.save_analysis(
            analysis_id=f"analysis-{index}",
            question=f"Pergunta de teste {index}",
            status="success",
            total_time_ms=100.0,
        )

    first_response = client.get(
        "/analyses/history",
        params={
            "limit": 2,
            "offset": 0,
        },
    )

    second_response = client.get(
        "/analyses/history",
        params={
            "limit": 2,
            "offset": 2,
        },
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    assert len(first_data["items"]) == 2
    assert len(second_data["items"]) == 1

    assert first_data["pagination"]["total"] == 3
    assert first_data["pagination"]["has_more"] is True

    assert second_data["pagination"]["offset"] == 2
    assert second_data["pagination"]["has_more"] is False


def test_analysis_history_filters_by_status(monkeypatch, tmp_path):
    """Verifica os filtros de sucesso e falha."""

    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    success_response = client.get(
        "/analyses/history",
        params={"status": "success"},
    )

    failure_response = client.get(
        "/analyses/history",
        params={"status": "failure"},
    )

    assert success_response.status_code == 200
    assert failure_response.status_code == 200

    success_data = success_response.json()
    failure_data = failure_response.json()

    assert success_data["pagination"]["total"] == 1
    assert failure_data["pagination"]["total"] == 1

    assert success_data["items"][0]["status"] == "success"
    assert failure_data["items"][0]["status"] == "failure"

    assert success_data["items"][0]["analysis_id"] == "analysis-success-001"

    assert failure_data["items"][0]["analysis_id"] == "analysis-failure-001"


def test_analysis_history_returns_complete_details(
    monkeypatch,
    tmp_path,
):
    """Verifica a recuperação dos dados completos de uma análise."""

    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    response = client.get("/analyses/history/analysis-success-001")

    assert response.status_code == 200

    data = response.json()

    assert data["analysis_id"] == "analysis-success-001"
    assert data["question"] == "Quantos pedidos foram entregues?"
    assert data["status"] == "success"

    assert data["total_time_ms"] == 1250.5
    assert data["cache_hit"] is False

    assert data["error_type"] is None
    assert data["error_message"] is None

    assert data["llm_metrics"]["provider"] == "gemini"
    assert data["llm_metrics"]["model"] == "gemini-2.5-flash"

    assert len(data["steps"]) == 2

    assert data["steps"][0]["name"] == "sql_generation"
    assert data["steps"][1]["name"] == "response_generation"

    assert data["created_at"] is not None


def test_analysis_history_returns_404_for_missing_analysis(
    monkeypatch,
    tmp_path,
):
    """Verifica o retorno 404 para análises inexistentes."""

    create_test_history(monkeypatch, tmp_path)

    response = client.get("/analyses/history/analysis-does-not-exist")

    assert response.status_code == 404

    assert response.json()["detail"] == ("Análise não encontrada.")


def test_analysis_history_rejects_invalid_status(
    monkeypatch,
    tmp_path,
):
    """Verifica a validação do filtro de status."""

    create_test_history(monkeypatch, tmp_path)

    response = client.get(
        "/analyses/history",
        params={"status": "pending"},
    )

    assert response.status_code == 400

    assert response.json()["detail"] == (
        "status deve ser 'success', 'failure' ou None."
    )


def test_analysis_history_rejects_invalid_pagination(
    monkeypatch,
    tmp_path,
):
    """Verifica os limites dos parâmetros de paginação."""

    create_test_history(monkeypatch, tmp_path)

    invalid_limit = client.get(
        "/analyses/history",
        params={"limit": 101},
    )

    invalid_offset = client.get(
        "/analyses/history",
        params={"offset": -1},
    )

    assert invalid_limit.status_code == 422
    assert invalid_offset.status_code == 422


def test_analysis_history_persists_after_reinitialization(
    monkeypatch,
    tmp_path,
):
    """Verifica a persistência após recriar a instância SQLite."""

    history = create_test_history(monkeypatch, tmp_path)

    history.save_analysis(
        analysis_id="persistent-analysis-001",
        question="Qual foi a taxa de cancelamento?",
        status="success",
        total_time_ms=1500.0,
        cache_hit=False,
        llm_metrics={
            "provider": "gemini",
            "model": "gemini-2.5-flash",
        },
        steps=[
            {
                "name": "sql_generation",
                "status": "completed",
            },
        ],
    )

    # Simula uma nova inicialização da aplicação.
    reopened_history = AnalysisHistory(history.database_path)

    monkeypatch.setattr(
        "ai_analyst.api.main.analysis_history",
        reopened_history,
    )

    response = client.get("/analyses/history/persistent-analysis-001")

    assert response.status_code == 200

    data = response.json()

    assert data["analysis_id"] == "persistent-analysis-001"
    assert data["question"] == "Qual foi a taxa de cancelamento?"
    assert data["status"] == "success"

    assert data["total_time_ms"] == 1500.0

    assert data["llm_metrics"]["provider"] == "gemini"

    assert data["steps"][0]["name"] == "sql_generation"


# ============================================================
# TESTES DE PERSISTÊNCIA DE FALHAS E RASTREABILIDADE
# ============================================================


@pytest.mark.parametrize(
    ("error_class", "expected_status"),
    [
        (LLMRateLimitError, 503),
        (LLMTimeoutError, 504),
        (LLMProviderError, 502),
    ],
)
def test_api_persists_failed_analyses(
    monkeypatch,
    tmp_path,
    error_class,
    expected_status,
):
    """
    Verifica se falhas do provedor são persistidas no SQLite
    e se o analysis_id retornado corresponde ao registro salvo.
    """

    history = create_test_history(monkeypatch, tmp_path)

    error_message = "Falha simulada para teste."

    question = "Qual foi o desempenho das entregas?"

    def simulate_failure(self, question):
        raise error_class(error_message)

    monkeypatch.setattr(
        BusinessAnalyst,
        "analyze",
        simulate_failure,
    )

    response = client.post(
        "/analyze",
        json={"question": question},
    )

    # Verifica o status HTTP.
    assert response.status_code == expected_status

    # Verifica a estrutura da resposta de erro.
    detail = response.json()["detail"]

    assert isinstance(detail, dict)
    assert detail["message"]
    assert detail["analysis_id"]

    # O identificador retornado precisa ser um UUID válido.
    response_analysis_id = detail["analysis_id"]
    UUID(response_analysis_id)

    # A tentativa precisa ter sido persistida.
    assert history.count_analyses() == 1

    # Recupera o registro pelo histórico.
    history_response = client.get(
        "/analyses/history",
        params={"status": "failure"},
    )

    assert history_response.status_code == 200

    history_data = history_response.json()

    assert history_data["pagination"]["total"] == 1
    assert len(history_data["items"]) == 1

    analysis = history_data["items"][0]

    # A resposta HTTP e o histórico devem usar o mesmo ID.
    assert analysis["analysis_id"] == response_analysis_id

    # Verifica os dados principais da execução.
    assert analysis["question"] == question
    assert analysis["status"] == "failure"
    assert analysis["cache_hit"] is False

    # Verifica as informações do erro.
    assert analysis["error_type"] == error_class.__name__
    assert analysis["error_message"] == error_message

    # Verifica as métricas individuais.
    assert analysis["llm_metrics"]["analysis_id"] == (response_analysis_id)

    # A falha acontece antes de uma análise ser concluída.
    assert analysis["steps"] == []

    # Confirma que o endpoint de detalhes recupera o mesmo registro.
    detail_response = client.get(f"/analyses/history/{response_analysis_id}")

    assert detail_response.status_code == 200

    saved_detail = detail_response.json()

    assert saved_detail["analysis_id"] == response_analysis_id
    assert saved_detail["status"] == "failure"
    assert saved_detail["error_type"] == error_class.__name__
    assert saved_detail["error_message"] == error_message


# ============================================================
# TESTES DE GERENCIAMENTO DO HISTORICO
# ============================================================


def test_analysis_history_can_be_renamed_and_pinned(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    rename = client.patch(
        "/analyses/history/analysis-success-001/title",
        json={"title": "Vendas entregues"},
    )

    pin = client.patch(
        "/analyses/history/analysis-success-001/pin",
        json={"enabled": True},
    )

    assert rename.status_code == 200
    assert rename.json()["title"] == "Vendas entregues"

    assert pin.status_code == 200
    assert pin.json()["is_pinned"] is True

    saved = history.get_analysis("analysis-success-001")

    assert saved["title"] == "Vendas entregues"
    assert saved["is_pinned"] is True


def test_locked_analysis_rejects_modifications(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    lock = client.patch(
        "/analyses/history/analysis-success-001/lock",
        json={"enabled": True},
    )

    assert lock.status_code == 200
    assert lock.json()["is_locked"] is True

    rename = client.patch(
        "/analyses/history/analysis-success-001/title",
        json={"title": "Novo titulo"},
    )

    pin = client.patch(
        "/analyses/history/analysis-success-001/pin",
        json={"enabled": True},
    )

    archive = client.patch(
        "/analyses/history/analysis-success-001/archive",
        json={"enabled": True},
    )

    delete = client.delete(
        "/analyses/history/analysis-success-001",
    )

    assert rename.status_code == 409
    assert pin.status_code == 409
    assert archive.status_code == 409
    assert delete.status_code == 409

    assert history.get_analysis("analysis-success-001") is not None


def test_locked_analysis_can_be_unlocked(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    client.patch(
        "/analyses/history/analysis-success-001/lock",
        json={"enabled": True},
    )

    response = client.patch(
        "/analyses/history/analysis-success-001/lock",
        json={"enabled": False},
    )

    assert response.status_code == 200
    assert response.json()["is_locked"] is False

    saved = history.get_analysis("analysis-success-001")
    assert saved["is_locked"] is False


def test_analysis_can_be_archived_and_restored(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    archive = client.patch(
        "/analyses/history/analysis-success-001/archive",
        json={"enabled": True},
    )

    assert archive.status_code == 200
    assert archive.json()["is_archived"] is True

    normal_history = client.get("/analyses/history")

    assert normal_history.status_code == 200
    assert all(
        item["analysis_id"] != "analysis-success-001"
        for item in normal_history.json()["items"]
    )

    archived_history = client.get(
        "/analyses/history",
        params={"include_archived": True},
    )

    assert archived_history.status_code == 200
    assert any(
        item["analysis_id"] == "analysis-success-001"
        for item in archived_history.json()["items"]
    )

    restore = client.patch(
        "/analyses/history/analysis-success-001/archive",
        json={"enabled": False},
    )

    assert restore.status_code == 200
    assert restore.json()["is_archived"] is False


def test_analysis_can_be_deleted(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    response = client.delete(
        "/analyses/history/analysis-failure-001",
    )

    assert response.status_code == 200
    assert history.get_analysis("analysis-failure-001") is None


def test_analysis_search_filters_title_and_question(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    client.patch(
        "/analyses/history/analysis-success-001/title",
        json={"title": "Relatorio mensal"},
    )

    by_title = client.get(
        "/analyses/history",
        params={"search": "Relatorio mensal"},
    )

    by_question = client.get(
        "/analyses/history",
        params={"search": "previsao de vendas"},
    )

    assert by_title.status_code == 200
    assert [item["analysis_id"] for item in by_title.json()["items"]] == [
        "analysis-success-001"
    ]

    assert by_question.status_code == 200
    assert [item["analysis_id"] for item in by_question.json()["items"]] == [
        "analysis-failure-001"
    ]


def test_analysis_projects_can_be_created_renamed_and_listed(
    monkeypatch,
    tmp_path,
):
    create_test_history(monkeypatch, tmp_path)

    create = client.post(
        "/analysis-projects",
        json={
            "name": "Vendas",
            "description": "Analises comerciais",
        },
    )

    assert create.status_code == 201

    project_id = create.json()["project_id"]

    rename = client.patch(
        f"/analysis-projects/{project_id}",
        json={"name": "Vendas 2026"},
    )

    assert rename.status_code == 200
    assert rename.json()["name"] == "Vendas 2026"

    listing = client.get("/analysis-projects")

    assert listing.status_code == 200
    assert any(
        project["project_id"] == project_id and project["name"] == "Vendas 2026"
        for project in listing.json()["items"]
    )


def test_analysis_can_be_moved_to_project(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    project = client.post(
        "/analysis-projects",
        json={"name": "Operacoes"},
    )

    assert project.status_code == 201

    project_id = project.json()["project_id"]

    move = client.patch(
        "/analyses/history/analysis-success-001/project",
        json={"project_id": project_id},
    )

    assert move.status_code == 200
    assert move.json()["project_id"] == project_id

    saved = history.get_analysis("analysis-success-001")
    assert saved["project_id"] == project_id

    filtered = client.get(
        "/analyses/history",
        params={"project_id": project_id},
    )

    assert filtered.status_code == 200
    assert [item["analysis_id"] for item in filtered.json()["items"]] == [
        "analysis-success-001"
    ]


def test_deleting_project_preserves_its_analyses(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    project = client.post(
        "/analysis-projects",
        json={"name": "Projeto temporario"},
    )

    project_id = project.json()["project_id"]

    client.patch(
        "/analyses/history/analysis-success-001/project",
        json={"project_id": project_id},
    )

    delete = client.delete(
        f"/analysis-projects/{project_id}",
    )

    assert delete.status_code == 200
    assert history.get_project(project_id) is None

    saved = history.get_analysis("analysis-success-001")

    assert saved is not None
    assert saved["project_id"] is None


def test_project_cannot_be_deleted_when_it_contains_locked_analysis(
    monkeypatch,
    tmp_path,
):
    history = create_test_history(monkeypatch, tmp_path)
    seed_test_history(history)

    project = client.post(
        "/analysis-projects",
        json={"name": "Projeto protegido"},
    )

    project_id = project.json()["project_id"]

    client.patch(
        "/analyses/history/analysis-success-001/project",
        json={"project_id": project_id},
    )

    client.patch(
        "/analyses/history/analysis-success-001/lock",
        json={"enabled": True},
    )

    delete = client.delete(
        f"/analysis-projects/{project_id}",
    )

    assert delete.status_code == 409
    assert history.get_project(project_id) is not None
    assert history.get_analysis("analysis-success-001") is not None


def test_api_history_management_endpoints_added():
    """Marcador para impedir a duplicacao deste bloco de testes."""
    assert True


def test_api_persists_steps_when_analysis_fails_during_execution(
    monkeypatch,
    tmp_path,
):
    """
    Preserva as etapas registradas antes de uma falha no BusinessAnalyst.
    """

    history = create_test_history(monkeypatch, tmp_path)

    def fail_database_version():
        raise RuntimeError("Falha simulada ao consultar a versão do banco.")

    monkeypatch.setattr(
        "ai_analyst.agent.business_analyst.get_database_version",
        fail_database_version,
    )

    response = client.post(
        "/analyze",
        json={"question": "Qual foi o desempenho das entregas?"},
    )

    assert response.status_code == 500

    analysis_id = response.json()["detail"]["analysis_id"]

    analysis = history.get_analysis(analysis_id)

    assert analysis is not None
    assert analysis["status"] == "failure"
    assert analysis["error_type"] == "RuntimeError"

    assert len(analysis["steps"]) == 1

    step = analysis["steps"][0]

    assert step["name"] == "database_version"
    assert step["status"] == "failed"
    assert step["duration_ms"] >= 0


# ============================================================
# TESTES DE PERSISTÊNCIA DE MÚLTIPLAS CONSULTAS
# ============================================================


def test_api_persists_multiple_queries_in_analysis_history(
    monkeypatch,
    tmp_path,
):
    """
    Verifica se a API persiste todas as consultas independentes
    e se o endpoint de detalhes recupera os dados integralmente.
    """

    history = create_test_history(monkeypatch, tmp_path)

    expected_queries = [
        {
            "objective": "Quantidade de pedidos por status",
            "sql": ("SELECT status, COUNT(*) AS total FROM orders GROUP BY status"),
            "status": "success",
            "results": [
                {
                    "status": "delivered",
                    "total": 120,
                },
            ],
            "total_results": 1,
            "chart": None,
            "error": None,
            "metrics": {
                "sql_generation_ms": 12.5,
                "sql_execution_ms": 3.2,
                "total_time_ms": 15.7,
            },
        },
        {
            "objective": "Valor médio dos pedidos",
            "sql": ("SELECT AVG(order_value) AS average_value FROM orders"),
            "status": "success",
            "results": [
                {
                    "average_value": 85.50,
                },
            ],
            "total_results": 1,
            "chart": None,
            "error": None,
            "metrics": {
                "sql_generation_ms": 10.0,
                "sql_execution_ms": 2.8,
                "total_time_ms": 12.8,
            },
        },
    ]

    def fake_analyze(self, question):
        return {
            "sql": expected_queries[0]["sql"],
            "results": expected_queries[0]["results"],
            "total_results": 1,
            "answer": "Análise concluída com duas consultas.",
            "chart": None,
            "steps": [
                {
                    "name": "query_planning",
                    "status": "completed",
                },
                {
                    "name": "multi_query_execution",
                    "status": "completed",
                },
            ],
            "queries": expected_queries,
            "cache_hit": False,
        }

    monkeypatch.setattr(
        BusinessAnalyst,
        "analyze",
        fake_analyze,
    )

    response = client.post(
        "/analyze",
        json={
            "question": (
                "Analise a quantidade de pedidos por status "
                "e o valor médio dos pedidos."
            ),
        },
    )

    assert response.status_code == 200

    response_data = response.json()
    analysis_id = response_data["analysis_id"]

    # A resposta da API deve conter as duas consultas.
    assert len(response_data["queries"]) == 2

    # A análise deve ter sido persistida.
    assert history.count_analyses() == 1

    saved_analysis = history.get_analysis(analysis_id)

    assert saved_analysis is not None
    assert saved_analysis["status"] == "success"

    # O SQLite deve preservar todas as consultas.
    assert saved_analysis["queries"] == expected_queries

    # O endpoint de detalhes deve recuperar os mesmos dados.
    detail_response = client.get(f"/analyses/history/{analysis_id}")

    assert detail_response.status_code == 200

    detail_data = detail_response.json()

    assert detail_data["analysis_id"] == analysis_id
    assert detail_data["status"] == "success"
    assert detail_data["queries"] == expected_queries

    assert len(detail_data["queries"]) == 2

    # Cada consulta precisa manter seu SQL independente.
    assert detail_data["queries"][0]["sql"] != detail_data["queries"][1]["sql"]

    assert all(query["status"] == "success" for query in detail_data["queries"])
