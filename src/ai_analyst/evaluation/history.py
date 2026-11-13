"""Persistência do histórico de benchmarks comparativos.

Armazena relatórios completos do BenchmarkRunner em SQLite,
mantendo metadados indexados para consultas históricas eficientes.

Cada benchmark é imutável após sua primeira persistência.
O identificador de uma execução não pode sobrescrever um registro
anterior, preservando a rastreabilidade dos experimentos.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_BENCHMARK_HISTORY_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "benchmark_history.sqlite3"
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkHistory:
    """Gerencia a persistência e consulta do histórico de benchmarks."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_BENCHMARK_HISTORY_PATH,
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize()

    # ==========================================================
    # CONEXÃO E INICIALIZAÇÃO
    # ==========================================================

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Abre, finaliza e fecha uma conexão SQLite.

        Cada operação possui sua própria conexão.

        Transações são confirmadas quando a operação termina
        normalmente e revertidas quando ocorre uma exceção.
        """

        connection = sqlite3.connect(
            str(self.database_path),
            timeout=10.0,
        )

        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA busy_timeout = 10000")
        connection.execute("PRAGMA foreign_keys = ON")

        try:
            yield connection

            if connection.in_transaction:
                connection.commit()

        except BaseException:
            if connection.in_transaction:
                connection.rollback()

            raise

        finally:
            connection.close()

    def initialize(self) -> None:
        """Cria as tabelas e os índices do histórico."""

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_history (
                    benchmark_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                        CHECK (schema_version >= 1),
                    dataset_version INTEGER NOT NULL
                        CHECK (dataset_version >= 1),
                    dataset_source TEXT,
                    dataset_case_count INTEGER NOT NULL
                        CHECK (dataset_case_count >= 1),
                    dataset_sha256 TEXT NOT NULL
                        CHECK (length(dataset_sha256) = 64),
                    providers_json TEXT NOT NULL,
                    provider_count INTEGER NOT NULL
                        CHECK (provider_count >= 2),
                    completed_providers INTEGER NOT NULL
                        CHECK (completed_providers >= 0),
                    failed_providers INTEGER NOT NULL
                        CHECK (failed_providers >= 0),
                    all_providers_completed INTEGER NOT NULL
                        CHECK (all_providers_completed IN (0, 1)),
                    report_json TEXT NOT NULL,
                    CHECK (
                        completed_providers + failed_providers = provider_count
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_provider_runs (
                    benchmark_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    run_id TEXT,
                    execution_status TEXT NOT NULL
                        CHECK (execution_status IN ('completed', 'error')),
                    FOREIGN KEY (benchmark_id)
                        REFERENCES benchmark_history(benchmark_id)
                        ON DELETE CASCADE,
                    PRIMARY KEY (benchmark_id, provider)
                )
                """
            )

            indexes = [
                (
                    "idx_benchmark_history_created_at",
                    "benchmark_history(created_at DESC, benchmark_id DESC)",
                ),
                (
                    "idx_benchmark_history_dataset_sha256",
                    "benchmark_history(dataset_sha256)",
                ),
                (
                    "idx_benchmark_provider",
                    "benchmark_provider_runs(provider, benchmark_id)",
                ),
                (
                    "idx_benchmark_model",
                    "benchmark_provider_runs(model, benchmark_id)",
                ),
                (
                    "idx_benchmark_provider_model",
                    "benchmark_provider_runs(provider, model, benchmark_id)",
                ),
            ]

            for index_name, columns in indexes:
                connection.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {columns}
                    """
                )

    # ==========================================================
    # VALIDAÇÃO
    # ==========================================================

    @staticmethod
    def _require_mapping(
        value: Any,
        field_name: str,
    ) -> Mapping[str, Any]:
        """Valida se um campo possui estrutura de mapeamento."""

        if not isinstance(value, Mapping):
            raise ValueError(
                f"O campo '{field_name}' deve ser um objeto de mapeamento."
            )

        return value

    @staticmethod
    def _require_string(
        value: Any,
        field_name: str,
    ) -> str:
        """Valida uma string obrigatória."""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"O campo '{field_name}' deve ser uma string não vazia.")

        return value

    @staticmethod
    def _require_integer(
        value: Any,
        field_name: str,
        *,
        minimum: int = 0,
    ) -> int:
        """Valida um inteiro sem aceitar booleanos."""

        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(
                f"O campo '{field_name}' deve ser um inteiro "
                f"maior ou igual a {minimum}."
            )

        return value

    @classmethod
    def _validate_benchmark_report(
        cls,
        report: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Valida o contrato do relatório consolidado.

        Verifica identificação, dataset, fingerprint, provedores,
        estados de execução e consistência das contagens.

        Não altera nem recalcula as métricas do relatório original.
        """

        benchmark_id = cls._require_string(
            report.get("benchmark_id"),
            "benchmark_id",
        )

        if len(benchmark_id) > 150:
            raise ValueError("benchmark_id deve ter no máximo 150 caracteres.")

        created_at = cls._require_string(
            report.get("created_at"),
            "created_at",
        )

        try:
            parsed_created_at = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("created_at deve ser uma data ISO 8601 válida.") from exc

        if parsed_created_at.tzinfo is None:
            raise ValueError("created_at deve possuir informação de fuso horário.")

        schema_version = cls._require_integer(
            report.get("schema_version"),
            "schema_version",
            minimum=1,
        )

        dataset = cls._require_mapping(
            report.get("dataset"),
            "dataset",
        )

        dataset_version = cls._require_integer(
            dataset.get("version"),
            "dataset.version",
            minimum=1,
        )

        case_count = cls._require_integer(
            dataset.get("case_count"),
            "dataset.case_count",
            minimum=1,
        )

        dataset_sha256 = cls._require_string(
            dataset.get("sha256"),
            "dataset.sha256",
        ).lower()

        if not SHA256_PATTERN.fullmatch(dataset_sha256):
            raise ValueError("dataset.sha256 deve ser um hash SHA-256 hexadecimal.")

        case_ids = dataset.get("case_ids")

        if (
            not isinstance(case_ids, list)
            or len(case_ids) != case_count
            or not all(
                isinstance(case_id, str) and bool(case_id.strip())
                for case_id in case_ids
            )
        ):
            raise ValueError(
                "dataset.case_ids deve conter exatamente "
                "um identificador válido por cenário."
            )

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "dataset.case_ids não pode conter identificadores duplicados."
            )

        dataset_source = dataset.get("source")

        if dataset_source is not None and not isinstance(dataset_source, str):
            raise ValueError("dataset.source deve ser uma string ou None.")

        configuration = cls._require_mapping(
            report.get("configuration"),
            "configuration",
        )

        providers = configuration.get("providers")

        if not isinstance(providers, list) or len(providers) < 2:
            raise ValueError(
                "configuration.providers deve conter pelo menos dois provedores."
            )

        normalized_providers: list[str] = []

        for provider in providers:
            provider_name = cls._require_string(
                provider,
                "configuration.providers",
            )

            normalized = provider_name.strip().lower()

            if normalized != provider_name:
                raise ValueError(
                    "Os provedores devem estar normalizados em minúsculas."
                )

            if normalized in normalized_providers:
                raise ValueError("configuration.providers contém duplicatas.")

            normalized_providers.append(normalized)

        provider_count = cls._require_integer(
            configuration.get("provider_count"),
            "configuration.provider_count",
            minimum=2,
        )

        scenario_count = cls._require_integer(
            configuration.get("scenario_count"),
            "configuration.scenario_count",
            minimum=1,
        )

        if provider_count != len(normalized_providers):
            raise ValueError(
                "configuration.provider_count não corresponde "
                "à quantidade de provedores."
            )

        if scenario_count != case_count:
            raise ValueError(
                "configuration.scenario_count não corresponde "
                "à quantidade de cenários do dataset."
            )

        provider_runs = cls._require_mapping(
            report.get("provider_runs"),
            "provider_runs",
        )

        if set(provider_runs) != set(normalized_providers):
            raise ValueError(
                "provider_runs deve conter exatamente os provedores configurados."
            )

        provider_records: list[dict[str, Any]] = []

        completed_count = 0
        failed_count = 0

        for provider in normalized_providers:
            execution = cls._require_mapping(
                provider_runs[provider],
                f"provider_runs.{provider}",
            )

            execution_status = execution.get("execution_status")

            if execution_status not in {"completed", "error"}:
                raise ValueError(
                    f"Estado de execução inválido para o provedor '{provider}'."
                )

            model = None
            run_id = None

            if execution_status == "completed":
                completed_count += 1

                individual_report = cls._require_mapping(
                    execution.get("report"),
                    f"provider_runs.{provider}.report",
                )

                model = individual_report.get("model")
                run_id = individual_report.get("run_id")

                if model is not None and not isinstance(model, str):
                    raise ValueError(
                        f"O modelo do provedor '{provider}' deve ser uma string."
                    )

                if run_id is not None and not isinstance(run_id, str):
                    raise ValueError(
                        f"O run_id do provedor '{provider}' deve ser uma string."
                    )

            else:
                failed_count += 1

            provider_records.append(
                {
                    "provider": provider,
                    "model": model,
                    "run_id": run_id,
                    "execution_status": execution_status,
                }
            )

        summary = cls._require_mapping(
            report.get("summary"),
            "summary",
        )

        summary_provider_count = cls._require_integer(
            summary.get("provider_count"),
            "summary.provider_count",
            minimum=2,
        )

        summary_completed = cls._require_integer(
            summary.get("completed_providers"),
            "summary.completed_providers",
        )

        summary_failed = cls._require_integer(
            summary.get("failed_providers"),
            "summary.failed_providers",
        )

        all_completed = summary.get("all_providers_completed")

        if not isinstance(all_completed, bool):
            raise ValueError("summary.all_providers_completed deve ser booleano.")

        if summary_provider_count != provider_count:
            raise ValueError("summary.provider_count não corresponde à configuração.")

        if summary_completed != completed_count or summary_failed != failed_count:
            raise ValueError(
                "As contagens de provedores no summary não correspondem "
                "às execuções individuais."
            )

        if completed_count + failed_count != provider_count:
            raise ValueError(
                "As execuções individuais possuem contagens inconsistentes."
            )

        if all_completed != (failed_count == 0):
            raise ValueError(
                "summary.all_providers_completed não corresponde "
                "aos estados individuais."
            )

        # Confirma que o relatório completo é serializável antes
        # de iniciar qualquer transação de gravação.
        try:
            report_json = json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "O relatório do benchmark não pode ser serializado como JSON."
            ) from exc

        providers_json = json.dumps(
            normalized_providers,
            ensure_ascii=False,
        )

        return {
            "benchmark_id": benchmark_id,
            "created_at": created_at,
            "schema_version": schema_version,
            "dataset_version": dataset_version,
            "dataset_source": dataset_source,
            "dataset_case_count": case_count,
            "dataset_sha256": dataset_sha256,
            "providers_json": providers_json,
            "provider_count": provider_count,
            "completed_providers": completed_count,
            "failed_providers": failed_count,
            "all_providers_completed": all_completed,
            "report_json": report_json,
            "provider_records": provider_records,
        }

    # ==========================================================
    # PERSISTÊNCIA
    # ==========================================================

    def save_benchmark(
        self,
        report: Mapping[str, Any],
    ) -> None:
        """Persiste um benchmark completo.

        O benchmark e seus metadados de provedores são inseridos
        na mesma transação.

        Um benchmark_id existente não pode ser sobrescrito.

        Raises:
            TypeError: Quando o relatório não é um mapeamento.
            ValueError: Quando o contrato do relatório é inválido
                ou o identificador já está armazenado.
        """

        if not isinstance(report, Mapping):
            raise TypeError("report deve ser um objeto de mapeamento.")

        data = self._validate_benchmark_report(report)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            existing = connection.execute(
                """
                SELECT benchmark_id
                FROM benchmark_history
                WHERE benchmark_id = ?
                """,
                (data["benchmark_id"],),
            ).fetchone()

            if existing is not None:
                raise ValueError(
                    f"O benchmark '{data['benchmark_id']}' já está armazenado."
                )

            connection.execute(
                """
                INSERT INTO benchmark_history (
                    benchmark_id,
                    created_at,
                    schema_version,
                    dataset_version,
                    dataset_source,
                    dataset_case_count,
                    dataset_sha256,
                    providers_json,
                    provider_count,
                    completed_providers,
                    failed_providers,
                    all_providers_completed,
                    report_json
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    data["benchmark_id"],
                    data["created_at"],
                    data["schema_version"],
                    data["dataset_version"],
                    data["dataset_source"],
                    data["dataset_case_count"],
                    data["dataset_sha256"],
                    data["providers_json"],
                    data["provider_count"],
                    data["completed_providers"],
                    data["failed_providers"],
                    int(data["all_providers_completed"]),
                    data["report_json"],
                ),
            )

            connection.executemany(
                """
                INSERT INTO benchmark_provider_runs (
                    benchmark_id,
                    provider,
                    model,
                    run_id,
                    execution_status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        data["benchmark_id"],
                        item["provider"],
                        item["model"],
                        item["run_id"],
                        item["execution_status"],
                    )
                    for item in data["provider_records"]
                ],
            )

    # ==========================================================
    # RECUPERAÇÃO DE RELATÓRIOS
    # ==========================================================

    @staticmethod
    def _deserialize_report(
        report_json: str,
        benchmark_id: str,
    ) -> dict[str, Any]:
        """Desserializa um relatório completo."""

        try:
            report = json.loads(report_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"O relatório do benchmark '{benchmark_id}' está corrompido."
            ) from exc

        if not isinstance(report, dict):
            raise ValueError(
                f"O relatório do benchmark '{benchmark_id}' possui formato inválido."
            )

        return report

    def get_benchmark(
        self,
        benchmark_id: str,
    ) -> dict[str, Any] | None:
        """Recupera o relatório completo de um benchmark."""

        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError("benchmark_id não pode ser vazio.")

        with self._connect() as connection:
            record = connection.execute(
                """
                SELECT report_json
                FROM benchmark_history
                WHERE benchmark_id = ?
                """,
                (benchmark_id,),
            ).fetchone()

        if record is None:
            return None

        return self._deserialize_report(
            record["report_json"],
            benchmark_id,
        )

    # ==========================================================
    # FILTROS
    # ==========================================================

    @staticmethod
    def _normalize_filter(
        value: str | None,
        field_name: str,
        *,
        lowercase: bool = False,
    ) -> str | None:
        """Valida e normaliza filtros textuais."""

        if value is None:
            return None

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"O filtro '{field_name}' não pode ser vazio.")

        normalized = value.strip()

        return normalized.lower() if lowercase else normalized

    @classmethod
    def _build_filter_clause(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        dataset_sha256: str | None = None,
    ) -> tuple[str, list[Any]]:
        """Constrói filtros parametrizados para consultas históricas."""

        normalized_provider = cls._normalize_filter(
            provider,
            "provider",
            lowercase=True,
        )

        normalized_model = cls._normalize_filter(
            model,
            "model",
        )

        normalized_fingerprint = cls._normalize_filter(
            dataset_sha256,
            "dataset_sha256",
            lowercase=True,
        )

        conditions: list[str] = []
        parameters: list[Any] = []

        if normalized_fingerprint is not None:
            if not SHA256_PATTERN.fullmatch(normalized_fingerprint):
                raise ValueError("dataset_sha256 deve ser um hash SHA-256 hexadecimal.")

            conditions.append("h.dataset_sha256 = ?")
            parameters.append(normalized_fingerprint)

        if normalized_provider is not None or normalized_model is not None:
            provider_conditions = [
                "p.benchmark_id = h.benchmark_id",
            ]

            if normalized_provider is not None:
                provider_conditions.append("p.provider = ? COLLATE NOCASE")
                parameters.append(normalized_provider)

            if normalized_model is not None:
                provider_conditions.append("p.model = ? COLLATE NOCASE")
                parameters.append(normalized_model)

            conditions.append(
                """
                EXISTS (
                    SELECT 1
                    FROM benchmark_provider_runs p
                    WHERE
                """
                + " AND ".join(provider_conditions)
                + ")"
            )

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        return where_clause, parameters

    # ==========================================================
    # RESUMO E LISTAGEM
    # ==========================================================

    @staticmethod
    def _deserialize_summary_record(
        record: sqlite3.Row,
        models: dict[str, str | None],
    ) -> dict[str, Any]:
        """Converte um registro para a representação resumida."""

        try:
            providers = json.loads(record["providers_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"A lista de provedores do benchmark "
                f"'{record['benchmark_id']}' está corrompida."
            ) from exc

        if not isinstance(providers, list):
            raise ValueError(
                f"A lista de provedores do benchmark "
                f"'{record['benchmark_id']}' possui formato inválido."
            )

        return {
            "benchmark_id": record["benchmark_id"],
            "created_at": record["created_at"],
            "schema_version": record["schema_version"],
            "dataset": {
                "version": record["dataset_version"],
                "source": record["dataset_source"],
                "case_count": record["dataset_case_count"],
                "sha256": record["dataset_sha256"],
            },
            "providers": providers,
            "models": models,
            "summary": {
                "provider_count": record["provider_count"],
                "completed_providers": record["completed_providers"],
                "failed_providers": record["failed_providers"],
                "all_providers_completed": bool(record["all_providers_completed"]),
            },
        }

    def list_benchmarks(
        self,
        limit: int = 20,
        offset: int = 0,
        provider: str | None = None,
        model: str | None = None,
        dataset_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Lista benchmarks resumidos com paginação e filtros.

        Os filtros podem ser combinados:
        - provider: provedor participante da execução.
        - model: modelo registrado na execução.
        - dataset_sha256: fingerprint do dataset.

        O relatório JSON completo não é carregado nesta operação.
        """

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
        ):
            raise ValueError("limit deve estar entre 1 e 100.")

        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset não pode ser negativo.")

        where_clause, parameters = self._build_filter_clause(
            provider=provider,
            model=model,
            dataset_sha256=dataset_sha256,
        )

        with self._connect() as connection:
            total = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM benchmark_history h
                {where_clause}
                """,
                parameters,
            ).fetchone()[0]

            records = connection.execute(
                f"""
                SELECT h.*
                FROM benchmark_history h
                {where_clause}
                ORDER BY
                    h.created_at DESC,
                    h.benchmark_id DESC
                LIMIT ? OFFSET ?
                """,
                [*parameters, limit, offset],
            ).fetchall()

            benchmark_ids = [record["benchmark_id"] for record in records]

            models_by_benchmark: dict[str, dict[str, str | None]] = {}

            if benchmark_ids:
                placeholders = ", ".join("?" for _ in benchmark_ids)

                provider_records = connection.execute(
                    f"""
                    SELECT
                        benchmark_id,
                        provider,
                        model
                    FROM benchmark_provider_runs
                    WHERE benchmark_id IN ({placeholders})
                    ORDER BY provider ASC
                    """,
                    benchmark_ids,
                ).fetchall()

                for provider_record in provider_records:
                    benchmark_id = provider_record["benchmark_id"]

                    models_by_benchmark.setdefault(
                        benchmark_id,
                        {},
                    )[provider_record["provider"]] = provider_record["model"]

        items = [
            self._deserialize_summary_record(
                record,
                models_by_benchmark.get(record["benchmark_id"], {}),
            )
            for record in records
        ]

        return {
            "items": items,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(records) < total,
            },
        }

    # ==========================================================
    # CONTAGEM
    # ==========================================================

    def count_benchmarks(
        self,
        provider: str | None = None,
        model: str | None = None,
        dataset_sha256: str | None = None,
    ) -> int:
        """Retorna a quantidade de benchmarks compatíveis com os filtros."""

        where_clause, parameters = self._build_filter_clause(
            provider=provider,
            model=model,
            dataset_sha256=dataset_sha256,
        )

        with self._connect() as connection:
            result = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM benchmark_history h
                {where_clause}
                """,
                parameters,
            ).fetchone()

        return int(result[0])

    # ==========================================================
    # EXCLUSÃO
    # ==========================================================

    def delete_benchmark(
        self,
        benchmark_id: str,
    ) -> bool:
        """Remove um benchmark e seus metadados de provedores.

        A exclusão dos registros relacionados é realizada
        automaticamente pela chave estrangeira ON DELETE CASCADE.
        """

        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError("benchmark_id não pode ser vazio.")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            cursor = connection.execute(
                """
                DELETE FROM benchmark_history
                WHERE benchmark_id = ?
                """,
                (benchmark_id,),
            )

        return cursor.rowcount > 0
