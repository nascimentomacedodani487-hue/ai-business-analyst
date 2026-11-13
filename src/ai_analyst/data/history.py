"""
Armazenamento persistente do histórico de análises.

Utiliza SQLite separado do DuckDB para registrar:
- Dados completos das análises.
- Múltiplas consultas SQL por análise.
- Títulos personalizados e configurações de organização.
- Fixação, bloqueio e arquivamento.
- Projetos e associação de análises.
- Pesquisa e paginação do histórico.

Cada operação utiliza sua própria conexão SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "analysis_history.sqlite3"
)


class AnalysisHistory:
    """Gerencia o histórico persistente das análises e projetos."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_HISTORY_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        """Normaliza texto para pesquisa sem distinção de acentos."""
        if value is None:
            return ""

        normalized = unicodedata.normalize("NFD", str(value))

        return "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        ).casefold()

    def _connect(self) -> sqlite3.Connection:
        """Abre uma conexão SQLite com configurações de segurança."""
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=10.0,
        )

        connection.row_factory = sqlite3.Row
        connection.create_function(
            "normalize_text",
            1,
            self._normalize_text,
            deterministic=True,
        )
        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")

        return connection

    def initialize(self) -> None:
        """Cria a estrutura e migra bancos existentes."""
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
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
                    error_message TEXT,
                    sql TEXT,
                    queries_json TEXT NOT NULL DEFAULT '[]',
                    results_json TEXT NOT NULL DEFAULT '[]',
                    total_results INTEGER NOT NULL DEFAULT 0,
                    answer TEXT,
                    chart_json TEXT,
                    provider TEXT,
                    model TEXT,
                    title TEXT,
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    is_locked INTEGER NOT NULL DEFAULT 0,
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    project_id TEXT REFERENCES analysis_projects(project_id)
                        ON DELETE SET NULL
                )
                """
            )

            existing_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(analysis_history)"
                ).fetchall()
            }

            migrations = {
                "sql": "TEXT",
                "queries_json": "TEXT NOT NULL DEFAULT '[]'",
                "results_json": "TEXT NOT NULL DEFAULT '[]'",
                "total_results": "INTEGER NOT NULL DEFAULT 0",
                "answer": "TEXT",
                "chart_json": "TEXT",
                "provider": "TEXT",
                "model": "TEXT",
                "title": "TEXT",
                "is_pinned": "INTEGER NOT NULL DEFAULT 0",
                "is_locked": "INTEGER NOT NULL DEFAULT 0",
                "is_archived": "INTEGER NOT NULL DEFAULT 0",
                "project_id": (
                    "TEXT REFERENCES analysis_projects(project_id) ON DELETE SET NULL"
                ),
            }

            for column, definition in migrations.items():
                if column not in existing_columns:
                    connection.execute(
                        f"""
                        ALTER TABLE analysis_history
                        ADD COLUMN {column} {definition}
                        """
                    )

            # Registros antigos recebem a pergunta como título inicial.
            connection.execute(
                """
                UPDATE analysis_history
                SET title = question
                WHERE title IS NULL OR TRIM(title) = ''
                """
            )

            indexes = [
                (
                    "idx_analysis_history_created_at",
                    "created_at DESC",
                ),
                (
                    "idx_analysis_history_status",
                    "status",
                ),
                (
                    "idx_analysis_history_provider",
                    "provider",
                ),
                (
                    "idx_analysis_history_pinned",
                    "is_pinned DESC, created_at DESC",
                ),
                (
                    "idx_analysis_history_project",
                    "project_id",
                ),
                (
                    "idx_analysis_history_archived",
                    "is_archived",
                ),
            ]

            for index_name, columns in indexes:
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON analysis_history({columns})
                    """
                )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_analysis_projects_name
                ON analysis_projects(name)
                """
            )

    @staticmethod
    def _serialize(value: Any) -> str:
        """Converte estruturas Python para JSON."""
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _deserialize_record(
        record: sqlite3.Row,
    ) -> dict[str, Any]:
        """Converte um registro SQLite completo para um dicionário."""
        return {
            "analysis_id": record["analysis_id"],
            "created_at": record["created_at"],
            "question": record["question"],
            "title": record["title"],
            "status": record["status"],
            "total_time_ms": record["total_time_ms"],
            "cache_hit": bool(record["cache_hit"]),
            "llm_metrics": json.loads(record["llm_metrics_json"]),
            "steps": json.loads(record["steps_json"]),
            "error_type": record["error_type"],
            "error_message": record["error_message"],
            "sql": record["sql"],
            "queries": json.loads(record["queries_json"]),
            "results": json.loads(record["results_json"]),
            "total_results": record["total_results"],
            "answer": record["answer"],
            "chart": (
                json.loads(record["chart_json"])
                if record["chart_json"] is not None
                else None
            ),
            "provider": record["provider"],
            "model": record["model"],
            "is_pinned": bool(record["is_pinned"]),
            "is_locked": bool(record["is_locked"]),
            "is_archived": bool(record["is_archived"]),
            "project_id": record["project_id"],
        }

    @staticmethod
    def _public_error_type(
        error_type: str | None,
    ) -> str | None:
        """
        Converte tipos técnicos de exceção em categorias públicas.

        Os nomes originais das classes permanecem armazenados no SQLite.
        Esta representação é destinada exclusivamente à exposição pela API.
        """

        if not isinstance(error_type, str) or not error_type.strip():
            return None

        public_types = {
            "LLMRateLimitError": "Limite de requisições da IA",
            "LLMTimeoutError": "Tempo limite da IA",
            "SQLExecutionTimeoutError": "Tempo limite da consulta SQL",
            "LLMTransientProviderError": "Indisponibilidade temporária da IA",
            "LLMProviderError": "Falha no serviço de IA",
            "SQLGuardrailError": "Falha na validação de segurança SQL",
            "SQLSyntaxError": "Falha na consulta SQL",
            "QueryPlanError": "Falha no planejamento",
            "UnsupportedAnalysisError": "Solicitação não suportada",
            "SQLIntentMismatchError": ("Incompatibilidade da consulta com o objetivo"),
            "ValueError": "Falha de validação",
        }

        return public_types.get(error_type, "Falha interna")

    @staticmethod
    def _summary_error_message(
        error_type: str | None,
    ) -> str:
        """
        Retorna uma mensagem controlada para exibição na API.

        A mensagem original da exceção permanece armazenada para
        diagnóstico, mas não é incluída na representação pública.
        """

        messages = {
            "LLMRateLimitError": (
                "O limite de requisições do serviço de IA foi atingido."
            ),
            "LLMTimeoutError": ("O serviço de IA excedeu o tempo máximo de resposta."),
            "SQLExecutionTimeoutError": (
                "A consulta SQL ultrapassou o tempo máximo de execução."
            ),
            "LLMTransientProviderError": (
                "O serviço de IA está temporariamente indisponível."
            ),
            "LLMProviderError": ("O serviço de IA apresentou uma falha."),
            "SQLGuardrailError": (
                "A consulta gerada não passou pelas validações de segurança."
            ),
            "SQLSyntaxError": ("A consulta SQL gerada não pôde ser executada."),
            "QueryPlanError": ("Não foi possível validar o planejamento da análise."),
            "UnsupportedAnalysisError": (
                "A solicitação não pode ser respondida com os dados disponíveis."
            ),
            "SQLIntentMismatchError": (
                "A consulta gerada não corresponde ao objetivo da análise."
            ),
            "ValueError": ("Não foi possível validar os dados da solicitação."),
        }

        safe_error_type = error_type if isinstance(error_type, str) else ""

        return messages.get(
            safe_error_type,
            (
                "A análise não pôde ser concluída. "
                "Os detalhes técnicos não são exibidos nesta listagem."
            ),
        )

    @staticmethod
    def _calculate_partial_success(
        queries: Any,
    ) -> bool:
        """Calcula se uma análise composta teve sucesso parcial."""

        if not isinstance(queries, list):
            return False

        successful_count = 0
        failed_count = 0
        unsupported_count = 0

        for query in queries:
            if not isinstance(query, dict):
                continue

            if query.get("status") == "success":
                successful_count += 1
                continue

            if query.get("status") != "failure":
                continue

            if query.get("failure_type") == "unsupported_capability":
                unsupported_count += 1
            else:
                failed_count += 1

        return successful_count > 0 and (unsupported_count > 0 or failed_count > 0)

    @staticmethod
    def _deserialize_summary_record(
        record: sqlite3.Row,
    ) -> dict[str, Any]:
        """
        Converte um registro para a representação resumida da listagem.

        Não devolve:
        - SQL executado.
        - Resultados completos.
        - Consultas independentes.
        - Métricas detalhadas do LLM.
        - Etapas internas.
        - Mensagens originais de exceções.
        - Nomes internos de classes de exceção.
        """

        try:
            queries = json.loads(record["queries_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            queries = []

        partial_success = AnalysisHistory._calculate_partial_success(queries)

        status = record["status"]
        original_error_type = record["error_type"]

        return {
            "analysis_id": record["analysis_id"],
            "created_at": record["created_at"],
            "question": record["question"],
            "title": record["title"],
            "status": status,
            "partial_success": partial_success,
            "total_time_ms": record["total_time_ms"],
            "cache_hit": bool(record["cache_hit"]),
            "error_type": AnalysisHistory._public_error_type(original_error_type),
            "error_message": (
                AnalysisHistory._summary_error_message(original_error_type)
                if status == "failure"
                else None
            ),
            "provider": record["provider"],
            "model": record["model"],
            "is_pinned": bool(record["is_pinned"]),
            "is_locked": bool(record["is_locked"]),
            "is_archived": bool(record["is_archived"]),
            "project_id": record["project_id"],
        }

    @staticmethod
    def _deserialize_project(
        record: sqlite3.Row,
    ) -> dict[str, Any]:
        """Converte um projeto SQLite para um dicionário."""
        return {
            "project_id": record["project_id"],
            "name": record["name"],
            "description": record["description"],
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "analysis_count": record["analysis_count"],
        }

    @staticmethod
    def _ensure_analysis_editable(
        connection: sqlite3.Connection,
        analysis_id: str,
    ) -> bool:
        """Verifica existência e bloqueio de uma análise."""
        record = connection.execute(
            """
            SELECT is_locked
            FROM analysis_history
            WHERE analysis_id = ?
            """,
            (analysis_id,),
        ).fetchone()

        if record is None:
            return False

        if record["is_locked"]:
            raise ValueError(
                "Esta análise está bloqueada. Desbloqueie-a antes de modificá-la."
            )

        return True

    def save_analysis(
        self,
        analysis_id: str,
        question: str,
        status: str,
        total_time_ms: float = 0.0,
        cache_hit: bool = False,
        llm_metrics: dict[str, Any] | None = None,
        steps: list[dict[str, Any]] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        sql: str | None = None,
        queries: list[dict[str, Any]] | None = None,
        results: list[dict[str, Any]] | None = None,
        total_results: int | None = None,
        answer: str | None = None,
        chart: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Persiste uma análise sem sobrescrever seus metadados pessoais."""
        if not analysis_id:
            raise ValueError("analysis_id não pode ser vazio.")

        if status not in {"success", "failure"}:
            raise ValueError("status deve ser 'success' ou 'failure'.")

        if total_time_ms < 0:
            raise ValueError("total_time_ms não pode ser negativo.")

        if queries is not None and not isinstance(queries, list):
            raise ValueError("queries deve ser uma lista.")

        if results is not None and not isinstance(results, list):
            raise ValueError("results deve ser uma lista.")

        if total_results is None:
            total_results = len(results or [])

        if total_results < 0:
            raise ValueError("total_results não pode ser negativo.")

        created_at = self._now()

        llm_metrics_json = self._serialize(llm_metrics or {})
        steps_json = self._serialize(steps or [])
        queries_json = self._serialize(queries or [])
        results_json = self._serialize(results or [])

        chart_json = self._serialize(chart) if chart is not None else None

        if error_message is not None:
            error_message = error_message[:500]

        with self._connect() as connection:
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
                    steps_json,
                    error_type,
                    error_message,
                    sql,
                    queries_json,
                    results_json,
                    total_results,
                    answer,
                    chart_json,
                    provider,
                    model,
                    title
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )

                ON CONFLICT(analysis_id)
                DO UPDATE SET
                    created_at = excluded.created_at,
                    question = excluded.question,
                    status = excluded.status,
                    total_time_ms = excluded.total_time_ms,
                    cache_hit = excluded.cache_hit,
                    llm_metrics_json = excluded.llm_metrics_json,
                    steps_json = excluded.steps_json,
                    error_type = excluded.error_type,
                    error_message = excluded.error_message,
                    sql = excluded.sql,
                    queries_json = excluded.queries_json,
                    results_json = excluded.results_json,
                    total_results = excluded.total_results,
                    answer = excluded.answer,
                    chart_json = excluded.chart_json,
                    provider = excluded.provider,
                    model = excluded.model
                """,
                (
                    analysis_id,
                    created_at,
                    question,
                    status,
                    total_time_ms,
                    int(cache_hit),
                    llm_metrics_json,
                    steps_json,
                    error_type,
                    error_message,
                    sql,
                    queries_json,
                    results_json,
                    total_results,
                    answer,
                    chart_json,
                    provider,
                    model,
                    question,
                ),
            )

    def get_analysis(
        self,
        analysis_id: str,
    ) -> dict[str, Any] | None:
        """Busca uma análise completa pelo identificador."""
        with self._connect() as connection:
            record = connection.execute(
                """
                SELECT *
                FROM analysis_history
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()

        if record is None:
            return None

        return self._deserialize_record(record)

    def _list_analyses(
        self,
        limit: int,
        offset: int,
        status: str | None,
        search: str | None,
        include_archived: bool,
        project_id: str | None,
        *,
        summary: bool,
    ) -> dict[str, Any]:
        """
        Executa a consulta comum de listagem.

        A seleção de colunas depende da representação solicitada,
        evitando carregar dados completos quando a listagem é resumida.
        """

        if not 1 <= limit <= 100:
            raise ValueError("limit deve estar entre 1 e 100.")

        if offset < 0:
            raise ValueError("offset não pode ser negativo.")

        if status is not None and status not in {
            "success",
            "failure",
        }:
            raise ValueError("status deve ser 'success', 'failure' ou None.")

        conditions: list[str] = []
        parameters: list[Any] = []

        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)

        if not include_archived:
            conditions.append("is_archived = 0")

        if search is not None and search.strip():
            conditions.append(
                "(normalize_text(title) LIKE ? OR normalize_text(question) LIKE ?)"
            )
            search_term = f"%{self._normalize_text(search.strip())}%"
            parameters.extend([search_term, search_term])

        if project_id == "__unassigned__":
            conditions.append("project_id IS NULL")
        elif project_id is not None:
            conditions.append("project_id = ?")
            parameters.append(project_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if summary:
            selected_columns = """
                analysis_id,
                created_at,
                question,
                title,
                status,
                total_time_ms,
                cache_hit,
                error_type,
                provider,
                model,
                is_pinned,
                is_locked,
                is_archived,
                project_id,
                queries_json
            """
        else:
            selected_columns = "*"

        with self._connect() as connection:
            total = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM analysis_history
                {where_clause}
                """,
                parameters,
            ).fetchone()[0]

            records = connection.execute(
                f"""
                SELECT {selected_columns}
                FROM analysis_history
                {where_clause}
                ORDER BY
                    is_pinned DESC,
                    created_at DESC,
                    analysis_id DESC
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()

        if summary:
            items = [self._deserialize_summary_record(record) for record in records]
        else:
            items = [self._deserialize_record(record) for record in records]

        return {
            "items": items,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(records) < total,
            },
        }

    def list_analyses(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        search: str | None = None,
        include_archived: bool = False,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Lista análises com seus dados completos.

        Mantido para compatibilidade com consumidores internos existentes.
        A API de listagem deve utilizar list_analysis_summaries().
        """

        return self._list_analyses(
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            include_archived=include_archived,
            project_id=project_id,
            summary=False,
        )

    def list_analysis_summaries(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str | None = None,
        search: str | None = None,
        include_archived: bool = False,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Lista análises sem carregar ou expor seus dados completos.

        Destinado aos endpoints de listagem e ao gerenciador de histórico.
        """

        return self._list_analyses(
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            include_archived=include_archived,
            project_id=project_id,
            summary=True,
        )

    def rename_analysis(
        self,
        analysis_id: str,
        title: str,
    ) -> bool:
        """Altera o título personalizado de uma análise."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("O título não pode ser vazio.")

        title = title.strip()

        if len(title) > 150:
            raise ValueError("O título deve ter no máximo 150 caracteres.")

        with self._connect() as connection:
            if not self._ensure_analysis_editable(
                connection,
                analysis_id,
            ):
                return False

            connection.execute(
                """
                UPDATE analysis_history
                SET title = ?
                WHERE analysis_id = ?
                """,
                (title, analysis_id),
            )

        return True

    def set_analysis_pinned(
        self,
        analysis_id: str,
        pinned: bool,
    ) -> bool:
        """Fixa ou remove a fixação de uma análise."""
        with self._connect() as connection:
            if not self._ensure_analysis_editable(
                connection,
                analysis_id,
            ):
                return False

            connection.execute(
                """
                UPDATE analysis_history
                SET is_pinned = ?
                WHERE analysis_id = ?
                """,
                (int(bool(pinned)), analysis_id),
            )

        return True

    def set_analysis_locked(
        self,
        analysis_id: str,
        locked: bool,
    ) -> bool:
        """Bloqueia ou desbloqueia uma análise."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_history
                SET is_locked = ?
                WHERE analysis_id = ?
                """,
                (int(bool(locked)), analysis_id),
            )

        return cursor.rowcount > 0

    def set_analysis_archived(
        self,
        analysis_id: str,
        archived: bool,
    ) -> bool:
        """Arquiva ou restaura uma análise."""
        with self._connect() as connection:
            if not self._ensure_analysis_editable(
                connection,
                analysis_id,
            ):
                return False

            connection.execute(
                """
                UPDATE analysis_history
                SET is_archived = ?
                WHERE analysis_id = ?
                """,
                (int(bool(archived)), analysis_id),
            )

        return True

    def move_analysis_to_project(
        self,
        analysis_id: str,
        project_id: str | None,
    ) -> bool:
        """Move uma análise para um projeto ou remove sua associação."""
        with self._connect() as connection:
            if not self._ensure_analysis_editable(
                connection,
                analysis_id,
            ):
                return False

            if project_id is not None:
                project = connection.execute(
                    """
                    SELECT project_id
                    FROM analysis_projects
                    WHERE project_id = ?
                    """,
                    (project_id,),
                ).fetchone()

                if project is None:
                    raise ValueError("Projeto não encontrado.")

            connection.execute(
                """
                UPDATE analysis_history
                SET project_id = ?
                WHERE analysis_id = ?
                """,
                (project_id, analysis_id),
            )

        return True

    def delete_analysis(
        self,
        analysis_id: str,
    ) -> bool:
        """Remove uma análise, desde que não esteja bloqueada."""
        with self._connect() as connection:
            if not self._ensure_analysis_editable(
                connection,
                analysis_id,
            ):
                return False

            cursor = connection.execute(
                """
                DELETE FROM analysis_history
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            )

        return cursor.rowcount > 0

    # ---------------------------------------------------------
    # Projetos
    # ---------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Cria um novo projeto."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("O nome do projeto não pode ser vazio.")

        name = name.strip()

        if len(name) > 100:
            raise ValueError("O nome do projeto deve ter no máximo 100 caracteres.")

        if description is not None:
            description = description.strip() or None

            if description and len(description) > 500:
                raise ValueError("A descrição não pode ter mais de 500 caracteres.")

        project_id = str(uuid.uuid4())
        now = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analysis_projects (
                    project_id,
                    name,
                    description,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    name,
                    description,
                    now,
                    now,
                ),
            )

        return self.get_project(project_id)  # type: ignore[return-value]

    def get_project(
        self,
        project_id: str,
    ) -> dict[str, Any] | None:
        """Busca um projeto pelo identificador."""
        with self._connect() as connection:
            record = connection.execute(
                """
                SELECT
                    p.*,
                    COUNT(a.analysis_id) AS analysis_count
                FROM analysis_projects p
                LEFT JOIN analysis_history a
                    ON a.project_id = p.project_id
                WHERE p.project_id = ?
                GROUP BY p.project_id
                """,
                (project_id,),
            ).fetchone()

        if record is None:
            return None

        return self._deserialize_project(record)

    def list_projects(self) -> list[dict[str, Any]]:
        """Lista projetos com a quantidade de análises associadas."""
        with self._connect() as connection:
            records = connection.execute(
                """
                SELECT
                    p.*,
                    COUNT(a.analysis_id) AS analysis_count
                FROM analysis_projects p
                LEFT JOIN analysis_history a
                    ON a.project_id = p.project_id
                GROUP BY p.project_id
                ORDER BY p.updated_at DESC, p.name ASC
                """
            ).fetchall()

        return [self._deserialize_project(record) for record in records]

    def rename_project(
        self,
        project_id: str,
        name: str,
    ) -> bool:
        """Renomeia um projeto."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("O nome do projeto não pode ser vazio.")

        name = name.strip()

        if len(name) > 100:
            raise ValueError("O nome do projeto deve ter no máximo 100 caracteres.")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_projects
                SET name = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (name, self._now(), project_id),
            )

        return cursor.rowcount > 0

    def delete_project(
        self,
        project_id: str,
    ) -> bool:
        """
        Exclui um projeto sem excluir suas análises.

        As análises associadas ficam sem projeto.
        Projetos com análises bloqueadas não podem ser excluídos.
        """
        with self._connect() as connection:
            project = connection.execute(
                """
                SELECT project_id
                FROM analysis_projects
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()

            if project is None:
                return False

            locked_analysis = connection.execute(
                """
                SELECT analysis_id
                FROM analysis_history
                WHERE project_id = ?
                    AND is_locked = 1
                LIMIT 1
                """,
                (project_id,),
            ).fetchone()

            if locked_analysis is not None:
                raise ValueError(
                    "O projeto contém análises bloqueadas. "
                    "Desbloqueie-as antes de excluir o projeto."
                )

            cursor = connection.execute(
                """
                DELETE FROM analysis_projects
                WHERE project_id = ?
                """,
                (project_id,),
            )

        return cursor.rowcount > 0

    def count_analyses(self) -> int:
        """Retorna a quantidade total de análises armazenadas."""
        with self._connect() as connection:
            result = connection.execute(
                """
                SELECT COUNT(*)
                FROM analysis_history
                """
            ).fetchone()

        return result[0]
