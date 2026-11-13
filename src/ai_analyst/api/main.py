"""
API principal do AI Business Analyst.
"""

import logging
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_analyst.agent.business_analyst import (
    DEFAULT_MODELS,
    BusinessAnalyst,
)
from ai_analyst.agent.cache import AnalysisCache
from ai_analyst.agent.capability_validator import (
    SQLIntentMismatchError,
    UnsupportedAnalysisError,
)
from ai_analyst.agent.query_planner import QueryPlanError
from ai_analyst.agent.sql_executor import SQLExecutionTimeoutError
from ai_analyst.data.db import connect_readonly
from ai_analyst.data.history import AnalysisHistory
from ai_analyst.data.schema import COLUMN_DOCS, TABLE_DOCS
from ai_analyst.llm.base import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ai_analyst.llm.metrics import llm_metrics

LOG_EXTRA_FIELDS = (
    "event",
    "request_id",
    "analysis_id",
    "method",
    "endpoint",
    "status_code",
    "provider",
    "model",
    "duration_ms",
    "prompt_length",
    "response_length",
    "prompt_tokens",
    "output_tokens",
    "thinking_tokens",
    "max_output_tokens",
    "error_type",
)

SAFE_CACHE_HIT_DETAILS = (
    "Análise recuperada do cache. Nenhuma chamada à IA foi realizada."
)

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if not any(
    isinstance(existing, logging.StreamHandler)
    and not isinstance(existing, logging.FileHandler)
    for existing in root_logger.handlers
):
    root_logger.addHandler(handler)


class DetailedFormatter(logging.Formatter):
    """Formata logs incluindo metadados adicionais."""

    def format(self, record):
        message = super().format(record)

        extra_fields = []

        for field in LOG_EXTRA_FIELDS:
            value = getattr(record, field, None)

            if value is not None:
                extra_fields.append(f"{field}={value}")

        if extra_fields:
            message += " | " + " | ".join(extra_fields)

        return message


for existing_handler in root_logger.handlers:
    if isinstance(existing_handler, logging.StreamHandler):
        existing_handler.setFormatter(
            DetailedFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )


logger = logging.getLogger(__name__)


analysis_cache = AnalysisCache(
    ttl_seconds=1800,
    max_entries=100,
)

analysis_history = AnalysisHistory()


app = FastAPI(
    title="AI Business Analyst API",
    description="API de análise de negócios utilizando inteligência artificial.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitora e registra todas as requisições HTTP."""

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:
        logger.exception(
            "Falha durante a requisição HTTP.",
            extra={
                "event": "http_request_failed",
                "request_id": request_id,
                "method": request.method,
                "endpoint": request.url.path,
            },
        )
        raise

    finally:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.info(
            "Requisição HTTP finalizada.",
            extra={
                "event": "http_request_completed",
                "request_id": request_id,
                "method": request.method,
                "endpoint": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )


class AnalysisRequest(BaseModel):
    """Dados recebidos para uma análise."""

    question: str = Field(
        min_length=3,
        max_length=50000,
        description="Pergunta de negócio em linguagem natural.",
    )

    provider: Literal[
        "gemini",
        "openai",
        "anthropic",
    ] = Field(
        default="gemini",
        description="Provedor de inteligência artificial utilizado.",
    )


class AnalysisTitleUpdate(BaseModel):
    """Novo título de uma análise."""

    title: str = Field(min_length=1, max_length=150)


class AnalysisBooleanUpdate(BaseModel):
    """Estado de uma propriedade booleana."""

    enabled: bool


class AnalysisProjectUpdate(BaseModel):
    """Projeto associado a uma análise."""

    project_id: str | None = None


class ProjectCreateRequest(BaseModel):
    """Dados para criação de um projeto."""

    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)


class ProjectRenameRequest(BaseModel):
    """Novo nome de um projeto."""

    name: str = Field(min_length=1, max_length=100)


def _log_analysis_exception(
    exc: Exception,
    *,
    analysis_id: str,
    request_id: str | None,
    provider: str,
    model: str,
    duration_ms: float,
) -> None:
    """
    Registra a exceção original e sua cadeia de causas.

    Não registra variáveis de ambiente nem credenciais.
    """

    original_exception = exc

    while original_exception.__cause__ is not None:
        original_exception = original_exception.__cause__

    logger.error(
        "Falha durante a análise de negócio.",
        exc_info=(
            type(exc),
            exc,
            exc.__traceback__,
        ),
        extra={
            "event": "analysis_failed",
            "analysis_id": analysis_id,
            "request_id": request_id,
            "provider": provider,
            "model": model,
            "duration_ms": duration_ms,
            "error_type": type(exc).__name__,
        },
    )

    if original_exception is not exc:
        logger.error(
            "Causa original da falha durante a análise.",
            exc_info=(
                type(original_exception),
                original_exception,
                original_exception.__traceback__,
            ),
            extra={
                "event": "analysis_original_exception",
                "analysis_id": analysis_id,
                "request_id": request_id,
                "provider": provider,
                "model": model,
                "error_type": type(original_exception).__name__,
            },
        )


@app.get("/health")
def health_check():
    """Verifica se a API está funcionando."""

    return {
        "status": "online",
        "service": "AI Business Analyst",
    }


@app.get("/metrics/llm")
def get_llm_metrics(
    provider: Literal[
        "gemini",
        "openai",
        "anthropic",
    ] = Query(default="gemini"),
):
    """Retorna métricas de utilização do provedor selecionado."""

    return llm_metrics.snapshot(provider=provider)


@app.get("/data/catalog")
def get_data_catalog():
    """
    Retorna o catálogo das tabelas do banco DuckDB.

    Inclui descrições, tipos das colunas,
    quantidade de registros e amostras.
    """

    connection = None

    try:
        connection = connect_readonly()

        tables = []

        for table_name, table_description in TABLE_DOCS.items():
            schema_rows = connection.execute(f"DESCRIBE {table_name}").fetchall()

            columns = []

            for column in schema_rows:
                column_name = column[0]
                column_type = column[1]
                nullable = column[2] == "YES"

                columns.append(
                    {
                        "name": column_name,
                        "type": column_type,
                        "nullable": nullable,
                        "description": COLUMN_DOCS.get(table_name, {}).get(
                            column_name,
                            "Sem descrição disponível.",
                        ),
                    }
                )

            total_rows = connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            sample_cursor = connection.execute(f"SELECT * FROM {table_name} LIMIT 5")

            sample_columns = [
                description[0] for description in sample_cursor.description
            ]

            sample_rows = sample_cursor.fetchall()

            sample = [dict(zip(sample_columns, row)) for row in sample_rows]

            tables.append(
                {
                    "name": table_name,
                    "description": table_description,
                    "total_rows": total_rows,
                    "columns": columns,
                    "sample": sample,
                }
            )

        return {
            "database": "DuckDB",
            "dataset": "Synthetic Delivery Dataset",
            "period": "2025",
            "tables_count": len(tables),
            "tables": tables,
        }

    except Exception as exc:
        logger.exception("Erro ao consultar o catálogo de dados.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível consultar o catálogo de dados.",
        ) from exc

    finally:
        if connection is not None:
            connection.close()


@app.post("/analyze")
def analyze_business(
    request: AnalysisRequest,
    http_request: Request,
):
    """
    Recebe uma pergunta, executa a análise e registra
    suas métricas e seu histórico individual.

    Permite selecionar Gemini, OpenAI ou Anthropic.
    O Gemini é utilizado quando nenhum provedor é informado.

    Todas as respostas operacionais de erro incluem o analysis_id.
    """

    analysis_id = str(uuid.uuid4())

    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    start_time = time.perf_counter()

    status = "failure"
    result = None
    analyst = None
    error_type = None
    error_message = None

    selected_provider = request.provider
    selected_model = DEFAULT_MODELS[selected_provider]

    # Rejeita perguntas contendo apenas espaços antes de inicializar o LLM.
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "A pergunta não pode estar vazia.",
                "analysis_id": analysis_id,
            },
        )

    with llm_metrics.track_analysis(analysis_id) as individual_metrics:
        try:
            analyst = BusinessAnalyst(
                cache=analysis_cache,
                provider_name=selected_provider,
            )

            selected_model = analyst.model

            result = analyst.analyze(request.question)

            selected_model = analyst.model
            status = "success"

            result["analysis_id"] = analysis_id
            result["provider"] = selected_provider
            result["model"] = selected_model
            result["llm_metrics"] = individual_metrics.snapshot()

            logger.info(
                "Análise de negócio concluída.",
                extra={
                    "event": "analysis_completed",
                    "analysis_id": analysis_id,
                    "request_id": request_id,
                    "provider": selected_provider,
                    "model": selected_model,
                    "duration_ms": round(
                        (time.perf_counter() - start_time) * 1000,
                        2,
                    ),
                },
            )

            return result

        except UnsupportedAnalysisError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            logger.warning(
                "Análise solicitada não suportada pelos dados disponíveis.",
                extra={
                    "event": "analysis_not_supported",
                    "analysis_id": analysis_id,
                    "request_id": request_id,
                    "provider": selected_provider,
                    "model": selected_model,
                    "error_type": error_type,
                },
            )

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "A solicitação não pode ser respondida "
                        "com os dados disponíveis."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except QueryPlanError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "Não foi possível interpretar o planejamento "
                        "gerado pelo serviço de IA."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except SQLIntentMismatchError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "Não foi possível gerar uma consulta "
                        "compatível com o objetivo solicitado."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except ValueError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Não foi possível concluir a análise.",
                    "analysis_id": analysis_id,
                },
            ) from exc

        except LLMRateLimitError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "O limite de requisições do serviço de IA "
                        "foi atingido. Tente novamente quando a cota "
                        "for renovada."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except LLMTimeoutError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=504,
                detail={
                    "message": (
                        "O serviço de inteligência artificial "
                        "demorou demais para responder."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except SQLExecutionTimeoutError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=504,
                detail={
                    "message": (
                        "A consulta SQL ultrapassou o tempo máximo "
                        "de execução permitido."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except LLMProviderError as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "O serviço de inteligência artificial apresentou um erro."
                    ),
                    "analysis_id": analysis_id,
                },
            ) from exc

        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

            _log_analysis_exception(
                exc,
                analysis_id=analysis_id,
                request_id=request_id,
                provider=selected_provider,
                model=selected_model,
                duration_ms=round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                ),
            )

            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Não foi possível concluir a análise.",
                    "analysis_id": analysis_id,
                },
            ) from exc

        finally:
            total_time_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            metrics_snapshot = individual_metrics.snapshot()

            if result is not None:
                steps_to_persist = result.get("steps", [])
                sql_to_persist = result.get("sql")

            elif analyst is not None:
                steps_to_persist = analyst.steps
                sql_to_persist = getattr(
                    analyst,
                    "last_sql",
                    None,
                )

            else:
                steps_to_persist = []
                sql_to_persist = None

            try:
                analysis_history.save_analysis(
                    analysis_id=analysis_id,
                    question=request.question,
                    status=status,
                    total_time_ms=total_time_ms,
                    cache_hit=(
                        result.get("cache_hit", False) if result is not None else False
                    ),
                    llm_metrics=metrics_snapshot,
                    steps=steps_to_persist,
                    error_type=error_type,
                    error_message=error_message,
                    sql=sql_to_persist,
                    queries=(result.get("queries", []) if result is not None else []),
                    results=(result.get("results", []) if result is not None else []),
                    total_results=(
                        result.get("total_results", 0) if result is not None else 0
                    ),
                    answer=(result.get("answer") if result is not None else None),
                    chart=(result.get("chart") if result is not None else None),
                    provider=selected_provider,
                    model=selected_model,
                )

            except Exception:
                logger.exception(
                    "Não foi possível persistir o histórico da análise.",
                    extra={
                        "event": "analysis_history_save_failed",
                        "analysis_id": analysis_id,
                        "request_id": request_id,
                    },
                )


def _require_analysis(analysis_id: str) -> dict:
    """Retorna uma análise ou gera 404."""

    result = analysis_history.get_analysis(analysis_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada.",
        )

    return result


def _public_analysis_detail(analysis: dict) -> dict:
    """
    Cria uma representação pública dos detalhes de uma análise.

    Preserva SQL, resultados, métricas e etapas, mas substitui
    mensagens técnicas de exceções por mensagens controladas.

    O registro original retornado pelo SQLite não é modificado.
    """

    public_analysis = dict(analysis)

    # Sanitiza o tipo e a mensagem da falha principal.
    original_error_type = public_analysis.get("error_type")

    if public_analysis.get("error_message") is not None:
        public_analysis["error_message"] = AnalysisHistory._summary_error_message(
            original_error_type,
        )

    if original_error_type is not None:
        public_analysis["error_type"] = AnalysisHistory._public_error_type(
            original_error_type,
        )

    # Sanitiza tipos e mensagens de falhas individuais das consultas.
    queries = public_analysis.get("queries")

    if isinstance(queries, list):
        public_queries = []

        for query in queries:
            if not isinstance(query, dict):
                public_queries.append(query)
                continue

            public_query = dict(query)
            original_query_error_type = public_query.get("error_type")

            if public_query.get("error") is not None:
                failure_type = public_query.get("failure_type")

                if failure_type == "unsupported_capability":
                    public_query["error"] = (
                        "Esta parte da análise não é compatível "
                        "com os recursos disponíveis."
                    )
                else:
                    public_query["error"] = AnalysisHistory._summary_error_message(
                        original_query_error_type,
                    )

            if original_query_error_type is not None:
                public_query["error_type"] = AnalysisHistory._public_error_type(
                    original_query_error_type,
                )

            public_queries.append(public_query)

        public_analysis["queries"] = public_queries

    # Sanitiza detalhes das etapas.
    steps = public_analysis.get("steps")

    if isinstance(steps, list):
        public_steps = []

        for step in steps:
            if not isinstance(step, dict):
                public_steps.append(step)
                continue

            public_step = dict(step)

            if public_step.get("details") is not None:
                step_name = public_step.get("name")
                step_status = public_step.get("status")

                if step_name == "cache_hit" and step_status == "completed":
                    public_step["details"] = SAFE_CACHE_HIT_DETAILS

                elif step_status == "unsupported":
                    public_step["details"] = (
                        "Esta etapa não é compatível com os dados disponíveis."
                    )

                elif step_status == "failed":
                    public_step["details"] = "Não foi possível concluir esta etapa."

                else:
                    public_step["details"] = "Detalhes técnicos omitidos."

            public_steps.append(public_step)

        public_analysis["steps"] = public_steps

    return public_analysis


def _mutation_error(exc: ValueError) -> HTTPException:
    """Converte erros de negócio em respostas HTTP."""

    message = str(exc)

    if "bloqueada" in message.lower():
        return HTTPException(
            status_code=409,
            detail=message,
        )

    if "projeto não encontrado" in message.lower():
        return HTTPException(
            status_code=404,
            detail=message,
        )

    return HTTPException(
        status_code=400,
        detail=message,
    )


@app.get("/analyses/history")
def get_analysis_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=150),
    include_archived: bool = Query(default=False),
    project_id: str | None = Query(default=None),
):
    """Lista análises com paginação, pesquisa e filtros."""

    try:
        return analysis_history.list_analysis_summaries(
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            include_archived=include_archived,
            project_id=project_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Erro ao consultar o histórico de análises.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível consultar o histórico.",
        ) from exc


@app.get("/analyses/history/{analysis_id}")
def get_analysis_history_detail(analysis_id: str):
    """Retorna os detalhes completos de uma análise específica."""

    try:
        analysis = _require_analysis(analysis_id)

        return _public_analysis_detail(analysis)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Erro ao consultar os detalhes da análise.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível consultar a análise.",
        ) from exc


@app.patch("/analyses/history/{analysis_id}/title")
def update_analysis_title(
    analysis_id: str,
    request: AnalysisTitleUpdate,
):
    """Renomeia uma análise."""

    _require_analysis(analysis_id)

    try:
        analysis_history.rename_analysis(
            analysis_id,
            request.title,
        )

        return {
            "message": "Análise renomeada com sucesso.",
            "analysis_id": analysis_id,
            "title": request.title.strip(),
        }

    except ValueError as exc:
        raise _mutation_error(exc) from exc


@app.patch("/analyses/history/{analysis_id}/pin")
def update_analysis_pin(
    analysis_id: str,
    request: AnalysisBooleanUpdate,
):
    """Fixa ou remove a fixação de uma análise."""

    _require_analysis(analysis_id)

    try:
        analysis_history.set_analysis_pinned(
            analysis_id,
            request.enabled,
        )

        return {
            "message": "Fixação atualizada com sucesso.",
            "analysis_id": analysis_id,
            "is_pinned": request.enabled,
        }

    except ValueError as exc:
        raise _mutation_error(exc) from exc


@app.patch("/analyses/history/{analysis_id}/lock")
def update_analysis_lock(
    analysis_id: str,
    request: AnalysisBooleanUpdate,
):
    """Bloqueia ou desbloqueia uma análise."""

    updated = analysis_history.set_analysis_locked(
        analysis_id,
        request.enabled,
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada.",
        )

    return {
        "message": (
            "Análise bloqueada com sucesso."
            if request.enabled
            else "Análise desbloqueada com sucesso."
        ),
        "analysis_id": analysis_id,
        "is_locked": request.enabled,
    }


@app.patch("/analyses/history/{analysis_id}/archive")
def update_analysis_archive(
    analysis_id: str,
    request: AnalysisBooleanUpdate,
):
    """Arquiva ou restaura uma análise."""

    _require_analysis(analysis_id)

    try:
        analysis_history.set_analysis_archived(
            analysis_id,
            request.enabled,
        )

        return {
            "message": (
                "Análise arquivada com sucesso."
                if request.enabled
                else "Análise restaurada com sucesso."
            ),
            "analysis_id": analysis_id,
            "is_archived": request.enabled,
        }

    except ValueError as exc:
        raise _mutation_error(exc) from exc


@app.patch("/analyses/history/{analysis_id}/project")
def update_analysis_project(
    analysis_id: str,
    request: AnalysisProjectUpdate,
):
    """Move uma análise para um projeto ou remove sua associação."""

    _require_analysis(analysis_id)

    try:
        analysis_history.move_analysis_to_project(
            analysis_id,
            request.project_id,
        )

        return {
            "message": "Projeto da análise atualizado com sucesso.",
            "analysis_id": analysis_id,
            "project_id": request.project_id,
        }

    except ValueError as exc:
        raise _mutation_error(exc) from exc


@app.delete("/analyses/history/{analysis_id}")
def remove_analysis(analysis_id: str):
    """Exclui uma análise desbloqueada."""

    _require_analysis(analysis_id)

    try:
        analysis_history.delete_analysis(analysis_id)

        return {
            "message": "Análise excluída com sucesso.",
            "analysis_id": analysis_id,
        }

    except ValueError as exc:
        raise _mutation_error(exc) from exc


# ---------------------------------------------------------
# Projetos
# ---------------------------------------------------------


@app.get("/analysis-projects")
def get_analysis_projects():
    """Lista projetos e suas quantidades de análises."""

    try:
        return {
            "items": analysis_history.list_projects(),
        }

    except Exception as exc:
        logger.exception("Erro ao consultar os projetos.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível consultar os projetos.",
        ) from exc


@app.post("/analysis-projects", status_code=201)
def create_analysis_project(request: ProjectCreateRequest):
    """Cria um projeto."""

    try:
        return analysis_history.create_project(
            name=request.name,
            description=request.description,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Erro ao criar projeto.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível criar o projeto.",
        ) from exc


@app.patch("/analysis-projects/{project_id}")
def update_analysis_project_name(
    project_id: str,
    request: ProjectRenameRequest,
):
    """Renomeia um projeto."""

    project = analysis_history.get_project(project_id)

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Projeto não encontrado.",
        )

    try:
        analysis_history.rename_project(
            project_id,
            request.name,
        )

        return {
            "message": "Projeto renomeado com sucesso.",
            "project_id": project_id,
            "name": request.name.strip(),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.delete("/analysis-projects/{project_id}")
def remove_analysis_project(project_id: str):
    """
    Exclui um projeto sem excluir suas análises.

    As análises associadas ficam sem projeto.
    """

    project = analysis_history.get_project(project_id)

    if project is None:
        raise HTTPException(
            status_code=404,
            detail="Projeto não encontrado.",
        )

    try:
        analysis_history.delete_project(project_id)

        return {
            "message": "Projeto excluído. As análises foram preservadas.",
            "project_id": project_id,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Erro ao excluir projeto.")

        raise HTTPException(
            status_code=500,
            detail="Não foi possível excluir o projeto.",
        ) from exc
