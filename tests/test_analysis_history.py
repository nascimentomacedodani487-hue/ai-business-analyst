
import json
import sqlite3

import pytest

from ai_analyst.data.history import AnalysisHistory


@pytest.fixture
def history(tmp_path):
    database_path = tmp_path / "test_history.sqlite3"
    return AnalysisHistory(database_path)


# =========================================================
# Testes básicos do histórico
# =========================================================

def test_initializes_database(history):
    assert history.count_analyses() == 0


def test_saves_and_retrieves_analysis(history):
    history.save_analysis(
        analysis_id="analysis-001",
        question="Qual a taxa de cancelamento?",
        status="success",
        total_time_ms=1250.5,
        cache_hit=False,
        llm_metrics={
            "requests": {"total": 2, "successful": 2},
            "tokens": {"total": 500},
        },
        steps=[
            {"name": "sql_generation", "duration_ms": 800}
        ],
    )

    result = history.get_analysis("analysis-001")

    assert result is not None
    assert result["question"] == "Qual a taxa de cancelamento?"
    assert result["status"] == "success"
    assert result["total_time_ms"] == 1250.5
    assert result["cache_hit"] is False
    assert result["llm_metrics"]["tokens"]["total"] == 500
    assert len(result["steps"]) == 1


def test_saves_and_retrieves_complete_analysis(history):
    analysis_data = {
        "analysis_id": "complete-001",
        "question": "Quais restaurantes tiveram mais pedidos?",
        "status": "success",
        "total_time_ms": 2450.75,
        "cache_hit": False,
        "llm_metrics": {
            "requests": {
                "total": 2,
                "successful": 2,
            },
            "tokens": {
                "input": 350,
                "output": 200,
                "total": 550,
            },
        },
        "steps": [
            {
                "name": "sql_generation",
                "duration_ms": 900,
            },
            {
                "name": "sql_execution",
                "duration_ms": 150,
            },
        ],
        "sql": (
            "SELECT restaurant_id, COUNT(*) AS total_orders "
            "FROM orders GROUP BY restaurant_id"
        ),
        "results": [
            {
                "restaurant_id": 10,
                "total_orders": 150,
            },
            {
                "restaurant_id": 20,
                "total_orders": 120,
            },
        ],
        "total_results": 2,
        "answer": "O restaurante 10 teve mais pedidos.",
        "chart": {
            "type": "bar",
            "title": "Pedidos por restaurante",
            "label_key": "restaurant_id",
            "value_key": "total_orders",
        },
    }

    history.save_analysis(**analysis_data)

    result = history.get_analysis("complete-001")

    assert result is not None
    assert result["analysis_id"] == "complete-001"
    assert result["question"] == analysis_data["question"]
    assert result["status"] == "success"
    assert result["total_time_ms"] == 2450.75
    assert result["cache_hit"] is False
    assert result["sql"] == analysis_data["sql"]
    assert result["results"] == analysis_data["results"]
    assert result["total_results"] == 2
    assert result["answer"] == analysis_data["answer"]
    assert result["chart"] == analysis_data["chart"]
    assert result["llm_metrics"] == analysis_data["llm_metrics"]
    assert result["steps"] == analysis_data["steps"]


def test_returns_none_for_unknown_analysis(history):
    assert history.get_analysis("inexistente") is None


def test_lists_analyses_with_pagination(history):
    for number in range(3):
        history.save_analysis(
            analysis_id=f"analysis-{number}",
            question=f"Pergunta {number}",
            status="success",
        )

    result = history.list_analyses(limit=2, offset=0)

    assert result["pagination"]["total"] == 3
    assert len(result["items"]) == 2
    assert result["pagination"]["has_more"] is True


def test_filters_analyses_by_status(history):
    history.save_analysis(
        analysis_id="success-001",
        question="Análise concluída",
        status="success",
    )

    history.save_analysis(
        analysis_id="failure-001",
        question="Análise com erro",
        status="failure",
        error_type="LLMProviderError",
        error_message="Falha no provedor",
    )

    result = history.list_analyses(status="failure")

    assert result["pagination"]["total"] == 1
    assert result["items"][0]["status"] == "failure"
    assert result["items"][0]["error_type"] == "LLMProviderError"


def test_persists_data_between_instances(tmp_path):
    database_path = tmp_path / "persistent_history.sqlite3"

    first = AnalysisHistory(database_path)
    first.save_analysis(
        analysis_id="persistent-001",
        question="Teste de persistência",
        status="success",
    )

    second = AnalysisHistory(database_path)

    assert second.count_analyses() == 1
    assert second.get_analysis("persistent-001") is not None


def test_migrates_existing_database_without_losing_records(tmp_path):
    database_path = tmp_path / "legacy_history.sqlite3"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE analysis_history (
                analysis_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('success', 'failure')),
                total_time_ms REAL NOT NULL DEFAULT 0,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                llm_metrics_json TEXT NOT NULL DEFAULT '{}',
                steps_json TEXT NOT NULL DEFAULT '[]',
                error_type TEXT,
                error_message TEXT
            )
            """
        )

        connection.execute(
            """
            INSERT INTO analysis_history (
                analysis_id,
                created_at,
                question,
                status,
                total_time_ms,
                cache_hit,
                llm_metrics_json,
                steps_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-001",
                "2026-01-01T10:00:00+00:00",
                "Análise antiga",
                "success",
                1200.0,
                0,
                json.dumps({"tokens": {"total": 100}}),
                json.dumps([]),
            ),
        )

    migrated_history = AnalysisHistory(database_path)

    result = migrated_history.get_analysis("legacy-001")

    assert result is not None
    assert result["analysis_id"] == "legacy-001"
    assert result["question"] == "Análise antiga"
    assert result["status"] == "success"
    assert result["total_time_ms"] == 1200.0
    assert result["llm_metrics"]["tokens"]["total"] == 100

    assert result["sql"] is None
    assert result["results"] == []
    assert result["total_results"] == 0
    assert result["answer"] is None
    assert result["chart"] is None

    # Novos metadados devem ser inicializados corretamente.
    assert result["title"] == "Análise antiga"
    assert result["is_pinned"] is False
    assert result["is_locked"] is False
    assert result["is_archived"] is False
    assert result["project_id"] is None

    assert migrated_history.count_analyses() == 1


def test_rejects_invalid_status(history):
    with pytest.raises(ValueError):
        history.save_analysis(
            analysis_id="invalid-001",
            question="Teste",
            status="pending",
        )


def test_rejects_invalid_pagination(history):
    with pytest.raises(ValueError):
        history.list_analyses(limit=0)

    with pytest.raises(ValueError):
        history.list_analyses(offset=-1)


def test_deletes_analysis(history):
    history.save_analysis(
        analysis_id="delete-001",
        question="Análise para excluir",
        status="success",
    )

    assert history.delete_analysis("delete-001") is True
    assert history.get_analysis("delete-001") is None
    assert history.delete_analysis("delete-001") is False


# =========================================================
# Testes de gerenciamento de análises
# =========================================================

def test_renames_analysis(history):
    history.save_analysis(
        analysis_id="rename-001",
        question="Qual a taxa de cancelamento?",
        status="success",
    )

    assert history.rename_analysis(
        "rename-001",
        "Taxa de cancelamento por região",
    ) is True

    result = history.get_analysis("rename-001")

    assert result["title"] == "Taxa de cancelamento por região"
    assert result["question"] == "Qual a taxa de cancelamento?"


def test_rejects_empty_analysis_title(history):
    history.save_analysis(
        analysis_id="rename-002",
        question="Teste",
        status="success",
    )

    with pytest.raises(ValueError):
        history.rename_analysis("rename-002", "   ")


def test_pinned_analyses_appear_first(history):
    history.save_analysis(
        analysis_id="normal-001",
        question="Análise normal",
        status="success",
    )

    history.save_analysis(
        analysis_id="pinned-001",
        question="Análise importante",
        status="success",
    )

    history.set_analysis_pinned("pinned-001", True)

    result = history.list_analyses()

    assert result["items"][0]["analysis_id"] == "pinned-001"
    assert result["items"][0]["is_pinned"] is True


def test_locked_analysis_rejects_modifications(history):
    history.save_analysis(
        analysis_id="locked-001",
        question="Análise protegida",
        status="success",
    )

    assert history.set_analysis_locked("locked-001", True) is True

    with pytest.raises(ValueError):
        history.rename_analysis("locked-001", "Novo título")

    with pytest.raises(ValueError):
        history.set_analysis_pinned("locked-001", True)

    with pytest.raises(ValueError):
        history.set_analysis_archived("locked-001", True)

    with pytest.raises(ValueError):
        history.delete_analysis("locked-001")

    result = history.get_analysis("locked-001")

    assert result["is_locked"] is True
    assert result["title"] == "Análise protegida"


def test_unlocks_analysis_and_allows_modifications(history):
    history.save_analysis(
        analysis_id="locked-002",
        question="Análise protegida",
        status="success",
    )

    history.set_analysis_locked("locked-002", True)
    history.set_analysis_locked("locked-002", False)

    assert history.rename_analysis(
        "locked-002",
        "Análise desbloqueada",
    ) is True


def test_archives_and_restores_analysis(history):
    history.save_analysis(
        analysis_id="archive-001",
        question="Análise arquivada",
        status="success",
    )

    assert history.set_analysis_archived(
        "archive-001",
        True,
    ) is True

    active = history.list_analyses()

    assert active["pagination"]["total"] == 0

    archived = history.list_analyses(include_archived=True)

    assert archived["pagination"]["total"] == 1
    assert archived["items"][0]["is_archived"] is True

    assert history.set_analysis_archived(
        "archive-001",
        False,
    ) is True

    restored = history.list_analyses()

    assert restored["pagination"]["total"] == 1


def test_searches_analysis_by_title_and_question(history):
    history.save_analysis(
        analysis_id="search-001",
        question="Quantos pedidos foram entregues?",
        status="success",
    )

    history.rename_analysis(
        "search-001",
        "Desempenho de entregas",
    )

    by_title = history.list_analyses(search="Desempenho")

    assert by_title["pagination"]["total"] == 1
    assert by_title["items"][0]["analysis_id"] == "search-001"

    by_question = history.list_analyses(
        search="pedidos foram entregues"
    )

    assert by_question["pagination"]["total"] == 1


# =========================================================
# Testes de projetos
# =========================================================

def test_creates_and_lists_projects(history):
    project = history.create_project(
        name="Análise comercial",
        description="Indicadores comerciais",
    )

    assert project["name"] == "Análise comercial"
    assert project["description"] == "Indicadores comerciais"
    assert project["analysis_count"] == 0

    projects = history.list_projects()

    assert len(projects) == 1
    assert projects[0]["project_id"] == project["project_id"]


def test_moves_analysis_between_projects(history):
    history.save_analysis(
        analysis_id="project-001",
        question="Análise de vendas",
        status="success",
    )

    project = history.create_project("Vendas")

    assert history.move_analysis_to_project(
        "project-001",
        project["project_id"],
    ) is True

    result = history.get_analysis("project-001")

    assert result["project_id"] == project["project_id"]

    project_details = history.get_project(project["project_id"])

    assert project_details["analysis_count"] == 1


def test_rejects_moving_analysis_to_unknown_project(history):
    history.save_analysis(
        analysis_id="project-002",
        question="Análise de vendas",
        status="success",
    )

    with pytest.raises(ValueError):
        history.move_analysis_to_project(
            "project-002",
            "projeto-inexistente",
        )


def test_deleting_project_preserves_analyses(history):
    history.save_analysis(
        analysis_id="project-003",
        question="Análise de vendas",
        status="success",
    )

    project = history.create_project("Projeto temporário")

    history.move_analysis_to_project(
        "project-003",
        project["project_id"],
    )

    assert history.delete_project(project["project_id"]) is True

    analysis = history.get_analysis("project-003")

    assert analysis is not None
    assert analysis["project_id"] is None


def test_cannot_delete_project_with_locked_analysis(history):
    history.save_analysis(
        analysis_id="project-004",
        question="Análise protegida",
        status="success",
    )

    project = history.create_project("Projeto protegido")

    history.move_analysis_to_project(
        "project-004",
        project["project_id"],
    )

    history.set_analysis_locked("project-004", True)

    with pytest.raises(ValueError):
        history.delete_project(project["project_id"])

    assert history.get_project(project["project_id"]) is not None


def test_save_analysis_preserves_custom_metadata(history):
    history.save_analysis(
        analysis_id="metadata-001",
        question="Pergunta original",
        status="success",
    )

    project = history.create_project("Projeto de teste")

    history.rename_analysis(
        "metadata-001",
        "Título personalizado",
    )
    history.set_analysis_pinned("metadata-001", True)
    history.set_analysis_archived("metadata-001", True)
    history.move_analysis_to_project(
        "metadata-001",
        project["project_id"],
    )

    history.save_analysis(
        analysis_id="metadata-001",
        question="Pergunta original atualizada",
        status="success",
        answer="Resposta atualizada",
    )

    result = history.get_analysis("metadata-001")

    assert result["title"] == "Título personalizado"
    assert result["is_pinned"] is True
    assert result["is_archived"] is True
    assert result["project_id"] == project["project_id"]
    assert result["question"] == "Pergunta original atualizada"
    assert result["answer"] == "Resposta atualizada"


def test_project_name_can_be_updated(history):
    project = history.create_project("Nome antigo")

    assert history.rename_project(
        project["project_id"],
        "Nome atualizado",
    ) is True

    updated = history.get_project(project["project_id"])

    assert updated["name"] == "Nome atualizado"


def test_rejects_empty_project_name(history):
    with pytest.raises(ValueError):
        history.create_project("   ")


def test_locked_analysis_cannot_be_moved(history):
    history.save_analysis(
        analysis_id="project-005",
        question="Análise bloqueada",
        status="success",
    )

    project = history.create_project("Novo projeto")

    history.set_analysis_locked("project-005", True)

    with pytest.raises(ValueError):
        history.move_analysis_to_project(
            "project-005",
            project["project_id"],
        )
