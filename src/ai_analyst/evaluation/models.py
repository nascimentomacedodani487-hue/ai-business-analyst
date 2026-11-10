"""Modelos de dados do AI Evaluation Engine.

Define as estruturas utilizadas para representar:
- Cenários de avaliação.
- Resultados individuais.
- Relatórios consolidados de execução.

Implementado com dataclasses para evitar dependências externas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ComparisonMode = Literal["exact", "tolerant"]

AnalysisMode = Literal[
    "single_query",
    "multi_query",
]

EvaluationStatus = Literal[
    "passed",
    "failed",
    "error",
    "skipped",
]


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Representa um cenário individual de avaliação.

    Attributes:
        case_id: Identificador único do cenário.
        question: Pergunta enviada ao agente.
        reference_sql: Consulta SQL confiável usada como referência.
        category: Categoria de negócio do cenário.
        description: Explicação do objetivo da avaliação.
        comparison_mode: Política de comparação dos valores.
        absolute_tolerance: Tolerância numérica absoluta.
        relative_tolerance: Tolerância numérica relativa.
        required_columns: Colunas obrigatórias no resultado.
        allow_extra_columns: Permite colunas adicionais.
        sort_rows: Ignora a ordem das linhas na comparação.
        tags: Etiquetas para agrupamento e filtragem.
        expected_analysis_mode: Modo de análise esperado pelo cenário.
            None permite qualquer modo.
            single_query exige uma única consulta.
            multi_query exige múltiplas consultas.
    """

    case_id: str
    question: str
    reference_sql: str
    category: str
    description: str = ""

    comparison_mode: ComparisonMode = "tolerant"

    absolute_tolerance: float = 0.01
    relative_tolerance: float = 0.0

    required_columns: tuple[str, ...] = ()

    allow_extra_columns: bool = False
    sort_rows: bool = True

    tags: tuple[str, ...] = ()

    expected_analysis_mode: AnalysisMode | None = None

    def __post_init__(self) -> None:
        """Valida os parâmetros do cenário."""

        required_text_fields = {
            "case_id": self.case_id,
            "question": self.question,
            "reference_sql": self.reference_sql,
            "category": self.category,
        }

        for field_name, value in required_text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"O campo '{field_name}' é obrigatório.")

        if self.comparison_mode not in ("exact", "tolerant"):
            raise ValueError("comparison_mode deve ser 'exact' ou 'tolerant'.")

        if self.expected_analysis_mode not in (None, "single_query", "multi_query"):
            raise ValueError(
                "expected_analysis_mode deve ser None, 'single_query' ou 'multi_query'."
            )

        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance não pode ser negativo.")

        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance não pode ser negativo.")

        if not isinstance(self.required_columns, tuple):
            object.__setattr__(
                self,
                "required_columns",
                tuple(self.required_columns),
            )

        if not isinstance(self.tags, tuple):
            object.__setattr__(
                self,
                "tags",
                tuple(self.tags),
            )


@dataclass(slots=True)
class EvaluationCaseResult:
    """Armazena o resultado da avaliação de um cenário.

    O status representa o resultado da execução:
    - passed: resultado aprovado.
    - failed: execução concluída, mas comparação reprovada.
    - error: erro durante a execução ou avaliação.
    - skipped: cenário não executado.
    """

    case_id: str
    question: str
    status: EvaluationStatus

    score: float = 0.0

    expected_rows: list[dict[str, Any]] = field(default_factory=list)

    actual_rows: list[dict[str, Any]] = field(default_factory=list)

    differences: list[str] = field(default_factory=list)

    analysis_id: str | None = None

    provider: str | None = None
    model: str | None = None

    duration_ms: float | None = None

    cache_hit: bool | None = None

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_statuses = {
            "passed",
            "failed",
            "error",
            "skipped",
        }

        if self.status not in valid_statuses:
            raise ValueError(f"Status de avaliação inválido: {self.status}")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score deve estar entre 0.0 e 1.0.")

        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms não pode ser negativo.")

    @property
    def passed(self) -> bool:
        """Indica se o cenário foi aprovado."""

        return self.status == "passed"

    @property
    def failed(self) -> bool:
        """Indica se a comparação foi reprovada."""

        return self.status == "failed"


@dataclass(slots=True)
class EvaluationRunReport:
    """Representa um relatório consolidado de avaliação."""

    run_id: str

    provider: str | None = None
    model: str | None = None

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    finished_at: datetime | None = None

    results: list[EvaluationCaseResult] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.status == "passed" for result in self.results)

    @property
    def failed_cases(self) -> int:
        return sum(result.status == "failed" for result in self.results)

    @property
    def error_cases(self) -> int:
        return sum(result.status == "error" for result in self.results)

    @property
    def skipped_cases(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def evaluated_cases(self) -> int:
        """Quantidade de cenários efetivamente avaliados."""

        return self.passed_cases + self.failed_cases + self.error_cases

    @property
    def pass_rate(self) -> float:
        """Taxa de aprovação em percentual.

        Cenários ignorados não entram no denominador.
        Erros de execução entram como reprovação.
        """

        if self.evaluated_cases == 0:
            return 0.0

        return round(
            self.passed_cases / self.evaluated_cases * 100,
            2,
        )

    @property
    def duration_ms(self) -> float:
        """Duração total da avaliação em milissegundos."""

        end_time = self.finished_at or datetime.now(timezone.utc)

        duration = (end_time - self.started_at).total_seconds() * 1000

        return round(max(duration, 0.0), 2)

    def finalize(self) -> None:
        """Registra o horário de conclusão da avaliação."""

        if self.finished_at is not None:
            raise RuntimeError("Este relatório já foi finalizado.")

        self.finished_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Converte o relatório para uma estrutura serializável."""

        return {
            "run_id": self.run_id,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at.isoformat(),
            "finished_at": (self.finished_at.isoformat() if self.finished_at else None),
            "summary": {
                "total_cases": self.total_cases,
                "evaluated_cases": self.evaluated_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "error_cases": self.error_cases,
                "skipped_cases": self.skipped_cases,
                "pass_rate": self.pass_rate,
                "duration_ms": self.duration_ms,
            },
            "results": [asdict(result) for result in self.results],
            "metadata": self.metadata,
        }
