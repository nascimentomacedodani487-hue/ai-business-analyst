"""
Avaliação determinística de resultados SQL, comportamentos e evidências.

Responsabilidades:
- Comparar resultados SQL de referência e resultados do agente.
- Aplicar tolerâncias numéricas absolutas e relativas.
- Tratar NaN, infinitos e valores nulos.
- Comparar linhas ordenadas e não ordenadas.
- Utilizar associação global de menor custo para linhas não ordenadas.
- Validar schema, colunas, linhas e células.
- Avaliar comportamentos esperados pelo Evaluation Engine.
- Verificar se SQL foi executado, inclusive em consultas com falha.
- Validar referências das afirmações às células dos resultados SQL.
- Identificar afirmações numéricas sem evidências correspondentes.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from math import isfinite, isnan
from typing import Any

from ai_analyst.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
)


class ResultEvaluator:
    """Executa avaliações determinísticas das análises do agente."""

    SUPPORTED_ANALYSIS_MODES = frozenset(
        {
            "single_query",
            "multi_query",
        }
    )

    BEHAVIOR_CRITERIA = frozenset(
        {
            "answer_present",
            "clarification_question_present",
            "explanation_present",
            "limitations_present",
            "supported_information_present",
            "sql_not_executed",
            "answer_evidence_present",
            "answer_evidence_valid",
            "answer_claims_supported",
        }
    )

    EVIDENCE_FIELDS = frozenset(
        {
            "query_id",
            "row_index",
            "column",
            "value",
        }
    )

    CLAIM_FIELDS = frozenset(
        {
            "claim",
            "evidence",
        }
    )

    NUMBER_PATTERN = re.compile(
        r"(?<![\w])"
        r"[+-]?"
        r"(?:\d{1,3}(?:\.\d{3})+|\d+)"
        r"(?:,\d+|\.\d+)?"
        r"%?"
        r"(?![\w])"
    )

    # ==========================================================
    # AVALIAÇÃO SQL
    # ==========================================================

    def evaluate(
        self,
        case: EvaluationCase,
        expected_rows: list[dict[str, Any]],
        actual_rows: list[dict[str, Any]],
        *,
        analysis_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        duration_ms: float | None = None,
        cache_hit: bool | None = None,
    ) -> EvaluationCaseResult:
        """Compara os resultados SQL com a referência confiável."""

        if case.evaluation_type != "sql_result":
            raise ValueError("evaluate() aceita somente cenários do tipo 'sql_result'.")

        self._validate_rows(expected_rows, "expected_rows")
        self._validate_rows(actual_rows, "actual_rows")

        expected = expected_rows
        actual = actual_rows

        differences: list[str] = []

        expected_columns = self._column_names(expected)
        actual_columns = self._column_names(actual)

        # ------------------------------------------------------
        # RESULTADOS VAZIOS
        # ------------------------------------------------------

        if not expected and not actual:
            return EvaluationCaseResult(
                case_id=case.case_id,
                question=case.question,
                status="passed",
                score=1.0,
                expected_rows=expected,
                actual_rows=actual,
                differences=[],
                analysis_id=analysis_id,
                provider=provider,
                model=model,
                duration_ms=duration_ms,
                cache_hit=cache_hit,
                metadata={
                    "schema_validation": "unavailable_empty_result",
                    "expected_row_count": 0,
                    "actual_row_count": 0,
                    "matched_rows": 0,
                    "fully_matched_rows": 0,
                    "row_accuracy": 1.0,
                    "cell_accuracy": 1.0,
                    "expected_columns": [],
                    "actual_columns": [],
                    "missing_columns": [],
                    "extra_columns": [],
                },
            )

        # ------------------------------------------------------
        # VALIDAÇÃO DE SCHEMA
        # ------------------------------------------------------

        required_columns = set(case.required_columns)

        missing_columns = (expected_columns | required_columns) - actual_columns

        extra_columns = actual_columns - expected_columns

        for column in sorted(missing_columns):
            differences.append(f"Coluna obrigatória ausente no resultado: '{column}'.")

        if not case.allow_extra_columns:
            for column in sorted(extra_columns):
                differences.append(f"Coluna inesperada no resultado: '{column}'.")

        schema_valid = not missing_columns and (
            case.allow_extra_columns or not extra_columns
        )

        comparison_columns = sorted(expected_columns | required_columns)

        # ------------------------------------------------------
        # COMPARAÇÃO DAS LINHAS
        # ------------------------------------------------------

        if case.sort_rows:
            assignments = self._minimum_cost_assignment(
                expected,
                actual,
                comparison_columns,
                case,
            )

        else:
            assignments = [
                (index, index) for index in range(min(len(expected), len(actual)))
            ]

        matched_rows = len(assignments)
        fully_matched_rows = 0

        compared_cells = 0
        matched_cells = 0

        unmatched_expected = set(range(len(expected)))
        unmatched_actual = set(range(len(actual)))

        for expected_index, actual_index in assignments:
            unmatched_expected.discard(expected_index)
            unmatched_actual.discard(actual_index)

            expected_row = expected[expected_index]
            actual_row = actual[actual_index]

            row_differences = self._compare_rows(
                expected_row,
                actual_row,
                comparison_columns,
                case,
                expected_index=expected_index,
                actual_index=actual_index,
            )

            total_row_cells = len(comparison_columns)
            row_mismatch_count = len(row_differences)

            compared_cells += total_row_cells

            matched_cells += max(
                total_row_cells - row_mismatch_count,
                0,
            )

            if not row_differences:
                fully_matched_rows += 1

            differences.extend(row_differences)

        # ------------------------------------------------------
        # LINHAS AUSENTES
        # ------------------------------------------------------

        for index in sorted(unmatched_expected):
            differences.append(
                f"Linha ausente no resultado: índice {index}, "
                f"registro {expected[index]!r}."
            )

            compared_cells += len(comparison_columns)

        # ------------------------------------------------------
        # LINHAS INESPERADAS
        # ------------------------------------------------------

        for index in sorted(unmatched_actual):
            differences.append(
                f"Linha inesperada no resultado: índice {index}, "
                f"registro {actual[index]!r}."
            )

        # ------------------------------------------------------
        # SCORE
        # ------------------------------------------------------

        denominator = max(
            len(expected),
            len(actual),
            1,
        )

        score = fully_matched_rows / denominator

        # Um schema incorreto nunca pode receber aprovação integral,
        # mesmo quando as células comuns apresentam os mesmos valores.
        if not schema_valid:
            score = min(score, 0.0)

        cell_accuracy = matched_cells / compared_cells if compared_cells else 1.0

        score = round(score, 6)
        cell_accuracy = round(cell_accuracy, 6)

        status = "passed" if not differences else "failed"

        metadata = {
            "schema_validation": ("passed" if schema_valid else "failed"),
            "expected_row_count": len(expected),
            "actual_row_count": len(actual),
            "matched_rows": matched_rows,
            "fully_matched_rows": fully_matched_rows,
            "missing_rows": len(unmatched_expected),
            "unexpected_rows": len(unmatched_actual),
            "row_accuracy": score,
            "cell_accuracy": cell_accuracy,
            "matched_cells": matched_cells,
            "compared_cells": compared_cells,
            "expected_columns": sorted(expected_columns),
            "actual_columns": sorted(actual_columns),
            "missing_columns": sorted(missing_columns),
            "extra_columns": sorted(extra_columns),
            "comparison_mode": case.comparison_mode,
            "absolute_tolerance": case.absolute_tolerance,
            "relative_tolerance": case.relative_tolerance,
            "sort_rows": case.sort_rows,
        }

        return EvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            status=status,
            score=score,
            expected_rows=expected,
            actual_rows=actual,
            differences=differences,
            analysis_id=analysis_id,
            provider=provider,
            model=model,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            metadata=metadata,
        )

    # ==========================================================
    # VALIDAÇÃO DOS ARGUMENTOS SQL
    # ==========================================================

    @staticmethod
    def _validate_rows(
        rows: Any,
        field_name: str,
    ) -> None:
        """Valida a estrutura das linhas recebidas."""

        if not isinstance(rows, list):
            raise TypeError(f"{field_name} deve ser uma lista.")

        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(
                    f"Cada registro de {field_name} deve ser um dicionário."
                )

            if any(not isinstance(column, str) for column in row):
                raise TypeError(
                    f"Todos os nomes do tipo string são obrigatórios "
                    f"em {field_name}, linha {row_index}."
                )

    @staticmethod
    def _column_names(
        rows: list[dict[str, Any]],
    ) -> set[str]:
        """Retorna todas as colunas presentes nas linhas."""

        return {column for row in rows for column in row}

    # ==========================================================
    # COMPARAÇÃO DE VALORES
    # ==========================================================

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Identifica valores numéricos sem incluir booleanos."""

        return isinstance(
            value,
            (int, float, Decimal),
        ) and not isinstance(value, bool)

    @staticmethod
    def _to_decimal(value: Any) -> Decimal:
        """Converte números em Decimal para comparação consistente."""

        if isinstance(value, Decimal):
            return value

        return Decimal(str(value))

    def _values_equal(
        self,
        expected: Any,
        actual: Any,
        case: EvaluationCase,
    ) -> bool:
        """Compara dois valores considerando a política de tolerância."""

        if expected is None or actual is None:
            return expected is None and actual is None

        # Booleanos não são considerados números.
        if isinstance(expected, bool) or isinstance(actual, bool):
            return (
                isinstance(expected, bool)
                and isinstance(actual, bool)
                and expected == actual
            )

        if self._is_numeric(expected) and self._is_numeric(actual):
            return self._numeric_values_equal(
                expected,
                actual,
                case,
            )

        return type(expected) is type(actual) and expected == actual

    def _numeric_values_equal(
        self,
        expected: int | float | Decimal,
        actual: int | float | Decimal,
        case: EvaluationCase,
    ) -> bool:
        """Aplica comparação exata ou tolerância numérica."""

        try:
            expected_decimal = self._to_decimal(expected)
            actual_decimal = self._to_decimal(actual)

        except (InvalidOperation, ValueError, TypeError):
            return False

        if expected_decimal.is_nan() or actual_decimal.is_nan():
            return expected_decimal.is_nan() and actual_decimal.is_nan()

        if expected_decimal.is_infinite() or actual_decimal.is_infinite():
            return expected_decimal == actual_decimal

        if case.comparison_mode == "exact":
            return expected_decimal == actual_decimal

        absolute_difference = abs(expected_decimal - actual_decimal)

        absolute_tolerance = Decimal(str(case.absolute_tolerance))

        relative_tolerance = Decimal(str(case.relative_tolerance))

        scale = max(
            abs(expected_decimal),
            abs(actual_decimal),
        )

        allowed_difference = max(
            absolute_tolerance,
            relative_tolerance * scale,
        )

        return absolute_difference <= allowed_difference

    @staticmethod
    def _format_value(value: Any) -> str:
        """Produz uma representação legível para diferenças."""

        return repr(value)

    def _compare_rows(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
        columns: list[str],
        case: EvaluationCase,
        *,
        expected_index: int,
        actual_index: int,
    ) -> list[str]:
        """Compara todas as células relevantes de duas linhas."""

        differences = []

        for column in columns:
            if column not in expected:
                differences.append(
                    f"Coluna '{column}' ausente na linha esperada {expected_index}."
                )
                continue

            if column not in actual:
                differences.append(
                    f"Coluna '{column}' ausente na linha obtida {actual_index}."
                )
                continue

            expected_value = expected[column]
            actual_value = actual[column]

            if self._values_equal(
                expected_value,
                actual_value,
                case,
            ):
                continue

            differences.append(
                f"Diferença na linha esperada {expected_index}, "
                f"linha obtida {actual_index}, coluna '{column}': "
                f"esperado {self._format_value(expected_value)}, "
                f"recebido {self._format_value(actual_value)}."
            )

        return differences

    # ==========================================================
    # ASSOCIAÇÃO GLOBAL DE LINHAS
    # ==========================================================

    def _minimum_cost_assignment(
        self,
        expected_rows: list[dict[str, Any]],
        actual_rows: list[dict[str, Any]],
        columns: list[str],
        case: EvaluationCase,
    ) -> list[tuple[int, int]]:
        """
        Associa linhas pela menor quantidade global de diferenças.

        Utiliza o algoritmo Húngaro para evitar associações gulosas
        que podem produzir um resultado global incorreto.
        """

        if not expected_rows or not actual_rows:
            return []

        cost_matrix = [
            [
                len(
                    self._compare_rows(
                        expected_row,
                        actual_row,
                        columns,
                        case,
                        expected_index=expected_index,
                        actual_index=actual_index,
                    )
                )
                for actual_index, actual_row in enumerate(actual_rows)
            ]
            for expected_index, expected_row in enumerate(expected_rows)
        ]

        return self._hungarian_assignment(cost_matrix)

    @classmethod
    def _hungarian_assignment(
        cls,
        costs: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Resolve a associação ótima de uma matriz retangular."""

        if not costs:
            return []

        if not costs[0]:
            return []

        row_count = len(costs)
        column_count = len(costs[0])

        if any(len(row) != column_count for row in costs):
            raise ValueError("A matriz de custos deve ser retangular.")

        if row_count > column_count:
            transposed = [
                [costs[row_index][column_index] for row_index in range(row_count)]
                for column_index in range(column_count)
            ]

            reversed_assignment = cls._hungarian_assignment(
                transposed,
            )

            return [
                (column_index, row_index)
                for row_index, column_index in reversed_assignment
            ]

        # Algoritmo Húngaro para número de linhas <= número de colunas.
        u = [0] * (row_count + 1)
        v = [0] * (column_count + 1)

        p = [0] * (column_count + 1)
        way = [0] * (column_count + 1)

        for row in range(1, row_count + 1):
            p[0] = row
            column0 = 0

            minimum_values = [float("inf") for _ in range(column_count + 1)]

            used = [False for _ in range(column_count + 1)]

            while True:
                used[column0] = True

                row0 = p[column0]
                delta = float("inf")
                column1 = 0

                for column in range(1, column_count + 1):
                    if used[column]:
                        continue

                    current = costs[row0 - 1][column - 1] - u[row0] - v[column]

                    if current < minimum_values[column]:
                        minimum_values[column] = current
                        way[column] = column0

                    if minimum_values[column] < delta:
                        delta = minimum_values[column]
                        column1 = column

                for column in range(column_count + 1):
                    if used[column]:
                        u[p[column]] += delta
                        v[column] -= delta
                    else:
                        minimum_values[column] -= delta

                column0 = column1

                if p[column0] == 0:
                    break

            while True:
                column1 = way[column0]
                p[column0] = p[column1]
                column0 = column1

                if column0 == 0:
                    break

        assignments = []

        for column in range(1, column_count + 1):
            if p[column] != 0:
                assignments.append(
                    (
                        p[column] - 1,
                        column - 1,
                    )
                )

        return sorted(assignments)

    # ==========================================================
    # AVALIAÇÃO COMPORTAMENTAL
    # ==========================================================

    def evaluate_behavior(
        self,
        case: EvaluationCase,
        analysis: Mapping[str, Any],
    ) -> EvaluationCaseResult:
        """Avalia o comportamento apresentado pelo agente."""

        if case.evaluation_type == "sql_result":
            raise ValueError(
                "evaluate_behavior() não aceita cenários do tipo 'sql_result'."
            )

        if not isinstance(analysis, Mapping):
            raise TypeError("analysis deve ser um objeto de mapeamento.")

        behavior = analysis.get("behavior")

        if not isinstance(behavior, Mapping):
            behavior = {}

        actual_behavior = behavior.get("type")
        expected_behavior = case.expected_behavior

        conditions: list[tuple[str, bool]] = []
        differences: list[str] = []

        behavior_matches = actual_behavior == expected_behavior

        conditions.append(
            (
                "expected_behavior",
                behavior_matches,
            )
        )

        if not behavior_matches:
            differences.append(
                f"Comportamento esperado '{expected_behavior}', "
                f"mas recebido '{actual_behavior}'."
            )

        evidence_report = None

        requested_evidence_criteria = {
            "answer_evidence_valid",
            "answer_claims_supported",
        }

        if (
            requested_evidence_criteria.intersection(case.evaluation_criteria)
            or "answer_evidence" in analysis
        ):
            evidence_report = self.validate_answer_evidence(
                analysis,
            )

        # ------------------------------------------------------
        # CRITÉRIOS
        # ------------------------------------------------------

        for criterion in case.evaluation_criteria:
            valid, message = self._evaluate_behavior_criterion(
                criterion,
                analysis,
                behavior,
                evidence_report,
            )

            conditions.append(
                (
                    criterion,
                    valid,
                )
            )

            if not valid:
                differences.append(message)

        passed_conditions = sum(passed for _, passed in conditions)

        total_conditions = len(conditions)

        score = passed_conditions / total_conditions if total_conditions else 0.0

        score = round(score, 6)

        status = "passed" if passed_conditions == total_conditions else "failed"

        metadata = {
            "evaluation_type": case.evaluation_type,
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
            "behavior_match": behavior_matches,
            "criteria_results": {name: passed for name, passed in conditions},
            "criteria_passed": passed_conditions,
            "criteria_total": total_conditions,
        }

        if evidence_report is not None:
            metadata.update(
                {
                    "answer_evidence_validation": evidence_report,
                    "answer_evidence_status": evidence_report.get("status"),
                    "answer_claims_supported": evidence_report.get("claims_supported"),
                }
            )

        return EvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            status=status,
            score=score,
            expected_rows=[],
            actual_rows=[],
            differences=differences,
            analysis_id=analysis.get("analysis_id"),
            provider=analysis.get("provider"),
            model=analysis.get("model"),
            duration_ms=self._optional_number(analysis.get("duration_ms")),
            cache_hit=analysis.get("cache_hit"),
            metadata=metadata,
        )

    def _evaluate_behavior_criterion(
        self,
        criterion: str,
        analysis: Mapping[str, Any],
        behavior: Mapping[str, Any],
        evidence_report: Mapping[str, Any] | None,
    ) -> tuple[bool, str]:
        """Avalia um critério comportamental individual."""

        if criterion not in self.BEHAVIOR_CRITERIA:
            return (
                False,
                f"Critério comportamental desconhecido: '{criterion}'.",
            )

        if criterion == "answer_present":
            answer = analysis.get("answer")

            valid = isinstance(answer, str) and bool(answer.strip())

            return (
                valid,
                "A resposta textual está ausente ou vazia.",
            )

        if criterion == "clarification_question_present":
            question = behavior.get("clarification_question")

            valid = isinstance(question, str) and bool(question.strip())

            return (
                valid,
                "A pergunta de esclarecimento está ausente ou vazia.",
            )

        if criterion == "explanation_present":
            reason = behavior.get("reason")

            valid = isinstance(reason, str) and bool(reason.strip())

            return (
                valid,
                "A justificativa comportamental está ausente ou vazia.",
            )

        if criterion == "limitations_present":
            limitations = behavior.get("limitations")

            valid = isinstance(limitations, list) and any(
                isinstance(item, str) and item.strip() for item in limitations
            )

            return (
                valid,
                "A decisão não apresenta limitações válidas.",
            )

        if criterion == "supported_information_present":
            information = behavior.get("supported_information")

            valid = isinstance(information, list) and any(
                isinstance(item, str) and item.strip() for item in information
            )

            return (
                valid,
                "A decisão não apresenta informações sustentadas.",
            )

        if criterion == "sql_not_executed":
            executed = self._was_sql_executed(
                analysis,
                behavior,
            )

            return (
                not executed,
                (
                    "Foi identificada execução SQL em um comportamento "
                    "que exige ausência de execução."
                ),
            )

        if criterion == "answer_evidence_present":
            evidence = analysis.get("answer_evidence")

            valid = isinstance(evidence, list) and len(evidence) > 0

            return (
                valid,
                "A resposta não apresenta evidências estruturadas.",
            )

        if criterion == "answer_evidence_valid":
            valid = (
                evidence_report is not None
                and evidence_report.get("status") == "verified"
            )

            return (
                valid,
                "As evidências da resposta não passaram na validação determinística.",
            )

        if criterion == "answer_claims_supported":
            valid = (
                evidence_report is not None
                and evidence_report.get("claims_supported") is True
            )

            return (
                valid,
                "Existem afirmações sem suporte verificável nos resultados SQL.",
            )

        return (
            False,
            f"Critério comportamental desconhecido: '{criterion}'.",
        )

    # ==========================================================
    # DETECÇÃO DE EXECUÇÃO SQL
    # ==========================================================

    @staticmethod
    def _was_sql_executed(
        analysis: Mapping[str, Any],
        behavior: Mapping[str, Any],
    ) -> bool:
        """
        Detecta execução SQL utilizando todas as evidências disponíveis.

        Uma consulta com status failure também pode ter sido executada.
        Por isso, o campo sql_executed de cada consulta é considerado.
        """

        if behavior.get("sql_executed") is True:
            return True

        if analysis.get("sql_executed") is True:
            return True

        successful_queries = analysis.get(
            "successful_queries",
            0,
        )

        if (
            isinstance(successful_queries, int)
            and not isinstance(successful_queries, bool)
            and successful_queries > 0
        ):
            return True

        queries = analysis.get("queries", [])

        if isinstance(queries, list):
            for query in queries:
                if not isinstance(query, Mapping):
                    continue

                if query.get("sql_executed") is True:
                    return True

                if query.get("status") == "success":
                    return True

        results = analysis.get("results")

        if isinstance(results, list) and results:
            return True

        return False

    # ==========================================================
    # VALIDAÇÃO DE EVIDÊNCIAS
    # ==========================================================

    def validate_answer_evidence(
        self,
        analysis: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Valida as afirmações estruturadas associadas à resposta.

        Verificações:
        - Estrutura dos claims.
        - Correspondência entre claims e texto da resposta.
        - Existência das consultas.
        - Sucesso das consultas referenciadas.
        - Existência de linhas e colunas.
        - Correspondência dos valores citados.
        - Cobertura dos números presentes na resposta.

        O resultado não representa uma prova semântica completa.
        Ele verifica a integridade das referências e a consistência
        numérica declarada pelo gerador.
        """

        if not isinstance(analysis, Mapping):
            raise TypeError("analysis deve ser um objeto de mapeamento.")

        answer_evidence = analysis.get("answer_evidence")

        if answer_evidence is None:
            return self._evidence_report(
                status="not_evaluated",
                differences=["A análise não possui answer_evidence."],
            )

        if not isinstance(answer_evidence, list):
            return self._evidence_report(
                status="failed",
                differences=["answer_evidence deve ser uma lista."],
            )

        answer = analysis.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            return self._evidence_report(
                status="failed",
                differences=["A análise não apresenta uma resposta textual válida."],
            )

        if not answer_evidence:
            numeric_values = self._extract_numeric_values(answer)

            if numeric_values:
                report = self._evidence_report(
                    status="failed",
                    differences=[
                        "A resposta apresenta valores numéricos, "
                        "mas não possui afirmações com evidências."
                    ],
                )

                report["answer_numeric_values"] = sorted(
                    str(value) for value in numeric_values
                )

                report["unsupported_numeric_values"] = sorted(
                    str(value) for value in numeric_values
                )

                return report

            return self._evidence_report(
                status="not_evaluated",
                differences=["Não existem afirmações estruturadas para validar."],
            )

        differences: list[str] = []

        query_map, query_errors = self._build_evidence_query_map(
            analysis,
        )

        differences.extend(query_errors)

        normalized_answer = self._normalize_claim_text(answer)
        padded_answer = f" {normalized_answer} "

        referenced_values: list[Any] = []
        all_claim_numbers: set[Decimal] = set()

        valid_claims = 0
        reference_count = 0
        valid_references = 0

        for claim_index, claim in enumerate(answer_evidence):
            if not isinstance(claim, Mapping):
                differences.append(
                    f"A afirmação {claim_index + 1} não é um objeto válido."
                )
                continue

            if set(claim) != self.CLAIM_FIELDS:
                differences.append(
                    f"A afirmação {claim_index + 1} possui campos inválidos."
                )
                continue

            claim_text = claim.get("claim")
            evidence = claim.get("evidence")

            if not isinstance(claim_text, str) or not claim_text.strip():
                differences.append(
                    f"A afirmação {claim_index + 1} possui texto vazio ou inválido."
                )
                continue

            if not isinstance(evidence, list) or not evidence:
                differences.append(
                    f"A afirmação {claim_index + 1} não possui evidências."
                )
                continue

            normalized_claim = self._normalize_claim_text(claim_text)

            # A correspondência exige limites de tokens.
            # Assim, a afirmação "receita 100" não é considerada
            # presente em uma resposta que contém apenas "receita 1000".
            claim_is_present = (
                bool(normalized_claim) and f" {normalized_claim} " in padded_answer
            )

            claim_valid = True

            if not claim_is_present:
                claim_valid = False

                differences.append(
                    f"A afirmação {claim_index + 1} não foi localizada "
                    "no texto da resposta."
                )

            claim_numbers = self._extract_numeric_values(claim_text)

            if claim_is_present:
                all_claim_numbers.update(claim_numbers)

            claim_referenced_values = []

            for evidence_index, reference in enumerate(evidence):
                reference_count += 1

                valid, value, error = self._validate_evidence_reference(
                    reference,
                    query_map,
                    claim_index=claim_index,
                    evidence_index=evidence_index,
                )

                if not valid:
                    claim_valid = False
                    differences.append(error)
                    continue

                valid_references += 1
                claim_referenced_values.append(value)

            # Uma afirmação ausente da resposta não pode fornecer
            # suporte numérico para os valores que aparecem nela.
            if claim_is_present:
                referenced_values.extend(claim_referenced_values)

            # Valores numéricos declarados em uma afirmação
            # devem estar representados nas células referenciadas.
            supported_claim_numbers: set[Decimal] = set()

            for value in claim_referenced_values:
                supported_claim_numbers.update(self._extract_value_numbers(value))

            unsupported_claim_numbers = claim_numbers - supported_claim_numbers

            if unsupported_claim_numbers:
                claim_valid = False

                formatted_numbers = sorted(
                    str(number) for number in unsupported_claim_numbers
                )

                differences.append(
                    f"A afirmação {claim_index + 1} contém valores "
                    "numéricos sem correspondência em suas evidências: "
                    f"{formatted_numbers}."
                )

            if claim_valid:
                valid_claims += 1

        # ------------------------------------------------------
        # COBERTURA NUMÉRICA DO TEXTO FINAL
        # ------------------------------------------------------

        answer_numbers = self._extract_numeric_values(answer)

        supported_answer_numbers: set[Decimal] = set()

        for value in referenced_values:
            supported_answer_numbers.update(self._extract_value_numbers(value))

        uncovered_answer_numbers = answer_numbers - all_claim_numbers

        if uncovered_answer_numbers:
            differences.append(
                "A resposta contém valores numéricos que não aparecem "
                "nas afirmações estruturadas: "
                f"{sorted(str(value) for value in uncovered_answer_numbers)}."
            )

        unsupported_answer_numbers = answer_numbers - supported_answer_numbers

        if unsupported_answer_numbers:
            differences.append(
                "A resposta contém valores numéricos sem correspondência "
                "nas células referenciadas: "
                f"{sorted(str(value) for value in unsupported_answer_numbers)}."
            )

        claims_supported = (
            len(answer_evidence) > 0
            and valid_claims == len(answer_evidence)
            and not differences
        )

        status = "verified" if claims_supported else "failed"

        return {
            "status": status,
            "claims_count": len(answer_evidence),
            "valid_claims": valid_claims,
            "invalid_claims": len(answer_evidence) - valid_claims,
            "reference_count": reference_count,
            "valid_references": valid_references,
            "invalid_references": reference_count - valid_references,
            "claims_supported": claims_supported,
            "answer_numeric_values": sorted(str(value) for value in answer_numbers),
            "unsupported_numeric_values": sorted(
                str(value) for value in unsupported_answer_numbers
            ),
            "uncovered_numeric_values": sorted(
                str(value) for value in uncovered_answer_numbers
            ),
            "differences": differences,
        }

    @staticmethod
    def _evidence_report(
        *,
        status: str,
        differences: list[str],
    ) -> dict[str, Any]:
        """Monta um relatório padronizado para ausência ou erro de evidências."""

        return {
            "status": status,
            "claims_count": 0,
            "valid_claims": 0,
            "invalid_claims": 0,
            "reference_count": 0,
            "valid_references": 0,
            "invalid_references": 0,
            "claims_supported": False,
            "answer_numeric_values": [],
            "unsupported_numeric_values": [],
            "uncovered_numeric_values": [],
            "differences": differences,
        }

    # ==========================================================
    # CONSTRUÇÃO DO MAPA DE CONSULTAS
    # ==========================================================

    def _build_evidence_query_map(
        self,
        analysis: Mapping[str, Any],
    ) -> tuple[dict[tuple[str, Any], dict[str, Any]], list[str]]:
        """Cria um mapa seguro das consultas que podem fundamentar claims.

        Análises multi_query exigem consultas individuais em
        analysis["queries"]. Resultados no nível raiz nunca são
        utilizados como fallback para esse tipo de análise.
        """

        query_map: dict[tuple[str, Any], dict[str, Any]] = {}
        errors: list[str] = []

        analysis_mode = analysis.get("analysis_mode")

        if analysis_mode is not None and (
            not isinstance(analysis_mode, str)
            or analysis_mode not in self.SUPPORTED_ANALYSIS_MODES
        ):
            return query_map, [
                f"Modo de análise inválido para validação de evidências: "
                f"{analysis_mode!r}."
            ]

        queries = analysis.get("queries")

        # ------------------------------------------------------
        # ANÁLISE COMPOSTA
        # ------------------------------------------------------

        if analysis_mode == "multi_query":
            if not isinstance(queries, list):
                return query_map, [
                    "Uma análise multi_query precisa apresentar "
                    "uma lista de consultas em 'queries'."
                ]

            if not queries:
                return query_map, [
                    "Uma análise multi_query não pode utilizar "
                    "resultados raiz como substitutos de consultas individuais."
                ]

        # ------------------------------------------------------
        # ANÁLISE COM CONSULTAS EXPLICITAMENTE DISPONÍVEIS
        # ------------------------------------------------------

        use_individual_queries = isinstance(queries, list) and bool(queries)

        if queries is not None and not isinstance(queries, list):
            return query_map, ["O campo 'queries' deve ser uma lista quando informado."]

        if use_individual_queries:
            for index, query in enumerate(queries, start=1):
                if not isinstance(query, Mapping):
                    errors.append(f"A consulta {index} possui estrutura inválida.")
                    continue

                query_id = query.get("id", index)

                if isinstance(query_id, bool) or not isinstance(query_id, (str, int)):
                    errors.append(f"A consulta {index} possui identificador inválido.")
                    continue

                if isinstance(query_id, str) and not query_id.strip():
                    errors.append(f"A consulta {index} possui identificador vazio.")
                    continue

                key = (
                    type(query_id).__name__,
                    query_id,
                )

                if key in query_map:
                    errors.append(f"Identificador duplicado de consulta: {query_id!r}.")
                    continue

                query_results = query.get("results", [])

                if not isinstance(query_results, list):
                    errors.append(
                        f"Os resultados da consulta {query_id!r} não são uma lista."
                    )
                    continue

                # Um resultado malformado invalida o mapa da consulta.
                rows_are_valid = True

                for row_index, row in enumerate(query_results):
                    if not isinstance(row, Mapping):
                        errors.append(
                            f"A linha {row_index} da consulta {query_id!r} "
                            "possui estrutura inválida."
                        )
                        rows_are_valid = False
                        continue

                    if any(not isinstance(column, str) for column in row):
                        errors.append(
                            f"A linha {row_index} da consulta {query_id!r} "
                            "possui nomes de coluna inválidos."
                        )
                        rows_are_valid = False

                if not rows_are_valid:
                    continue

                truncated_cells = query.get(
                    "truncated_cells",
                    [],
                )

                if not isinstance(truncated_cells, list):
                    errors.append(
                        f"As células truncadas da consulta {query_id!r} "
                        "devem ser uma lista."
                    )
                    continue

                truncated_cells_are_valid = True

                for cell_index, cell in enumerate(truncated_cells):
                    if not isinstance(cell, Mapping):
                        errors.append(
                            f"A célula truncada {cell_index} da consulta "
                            f"{query_id!r} possui estrutura inválida."
                        )
                        truncated_cells_are_valid = False
                        continue

                    cell_row_index = cell.get("row_index")
                    cell_column = cell.get("column")

                    if (
                        isinstance(cell_row_index, bool)
                        or not isinstance(cell_row_index, int)
                        or cell_row_index < 0
                        or not isinstance(cell_column, str)
                        or not cell_column.strip()
                    ):
                        errors.append(
                            f"A célula truncada {cell_index} da consulta "
                            f"{query_id!r} possui localização inválida."
                        )
                        truncated_cells_are_valid = False

                if not truncated_cells_are_valid:
                    continue

                query_map[key] = {
                    "id": query_id,
                    "status": query.get("status"),
                    "results": query_results,
                    "truncated_cells": truncated_cells,
                }

            return query_map, errors

        # ------------------------------------------------------
        # CONSULTA SIMPLES
        #
        # No fluxo simples do BusinessAnalyst, as referências
        # utilizam query_id=1 e os resultados ficam no nível raiz.
        #
        # Esse fallback só é permitido quando a análise não
        # foi identificada como multi_query.
        # ------------------------------------------------------

        results = analysis.get("results", [])

        if not isinstance(results, list):
            errors.append("Os resultados principais da análise não são uma lista.")
            return query_map, errors

        for row_index, row in enumerate(results):
            if not isinstance(row, Mapping):
                errors.append(
                    f"A linha {row_index} dos resultados principais "
                    "possui estrutura inválida."
                )
                return query_map, errors

            if any(not isinstance(column, str) for column in row):
                errors.append(
                    f"A linha {row_index} dos resultados principais "
                    "possui nomes de coluna inválidos."
                )
                return query_map, errors

        behavior = analysis.get("behavior")
        behavior = behavior if isinstance(behavior, Mapping) else {}

        successful_queries = analysis.get(
            "successful_queries",
            0,
        )

        successful_query_count = (
            isinstance(successful_queries, int)
            and not isinstance(successful_queries, bool)
            and successful_queries > 0
        )

        sql_executed = (
            behavior.get("sql_executed") is True
            or analysis.get("sql_executed") is True
            or successful_query_count
            or bool(results)
        )

        query_map[("int", 1)] = {
            "id": 1,
            "status": ("success" if sql_executed else "failure"),
            "results": results,
            "truncated_cells": [],
        }

        return query_map, errors

    # ==========================================================
    # VALIDAÇÃO DE UMA REFERÊNCIA
    # ==========================================================

    def _validate_evidence_reference(
        self,
        reference: Any,
        query_map: dict[tuple[str, Any], dict[str, Any]],
        *,
        claim_index: int,
        evidence_index: int,
    ) -> tuple[bool, Any, str]:
        """Valida uma única referência de evidência."""

        location = f"afirmação {claim_index + 1}, evidência {evidence_index + 1}"

        if not isinstance(reference, Mapping):
            return (
                False,
                None,
                f"A referência da {location} não é um objeto.",
            )

        if set(reference) != self.EVIDENCE_FIELDS:
            return (
                False,
                None,
                f"A referência da {location} possui campos inválidos.",
            )

        query_id = reference.get("query_id")
        row_index = reference.get("row_index")
        column = reference.get("column")
        cited_value = reference.get("value")

        if isinstance(query_id, bool) or not isinstance(query_id, (str, int)):
            return (
                False,
                None,
                f"O identificador da consulta da {location} é inválido.",
            )

        if isinstance(query_id, str) and not query_id.strip():
            return (
                False,
                None,
                f"O identificador da consulta da {location} está vazio.",
            )

        if isinstance(row_index, bool) or not isinstance(row_index, int):
            return (
                False,
                None,
                f"O índice da linha da {location} é inválido.",
            )

        if row_index < 0:
            return (
                False,
                None,
                f"O índice da linha da {location} é negativo.",
            )

        if not isinstance(column, str) or not column.strip():
            return (
                False,
                None,
                f"A coluna da {location} é inválida.",
            )

        if not self._is_json_scalar(cited_value):
            return (
                False,
                None,
                f"O valor da {location} não é um escalar JSON válido.",
            )

        query = query_map.get(
            (
                type(query_id).__name__,
                query_id,
            )
        )

        if query is None:
            return (
                False,
                None,
                f"A consulta referenciada na {location} não existe.",
            )

        if query["status"] != "success":
            return (
                False,
                None,
                f"A consulta referenciada na {location} não foi concluída.",
            )

        results = query["results"]

        if row_index >= len(results):
            return (
                False,
                None,
                f"A linha referenciada na {location} não existe.",
            )

        row = results[row_index]

        if not isinstance(row, Mapping):
            return (
                False,
                None,
                f"A linha referenciada na {location} é inválida.",
            )

        if column not in row:
            return (
                False,
                None,
                f"A coluna referenciada na {location} não existe.",
            )

        truncated_cells = query.get(
            "truncated_cells",
            [],
        )

        for cell in truncated_cells:
            if not isinstance(cell, Mapping):
                continue

            cell_row_index = cell.get("row_index")

            if (
                isinstance(cell_row_index, int)
                and not isinstance(cell_row_index, bool)
                and cell_row_index == row_index
                and cell.get("column") == column
            ):
                return (
                    False,
                    None,
                    f"A célula referenciada na {location} foi truncada.",
                )

        actual_value = row[column]

        if isinstance(
            actual_value,
            (Mapping, list, tuple, set),
        ):
            return (
                False,
                None,
                f"A célula referenciada na {location} não é escalar.",
            )

        normalized_value = self._json_compatible_value(
            actual_value,
        )

        if type(normalized_value) is not type(
            cited_value
        ) or not self._scalar_values_equal(
            normalized_value,
            cited_value,
        ):
            return (
                False,
                None,
                f"O valor informado na {location} não corresponde "
                "ao resultado original.",
            )

        return True, normalized_value, ""

    # ==========================================================
    # NORMALIZAÇÃO DOS VALORES DAS EVIDÊNCIAS
    # ==========================================================

    @staticmethod
    def _is_json_scalar(value: Any) -> bool:
        """Verifica se o valor pode ser representado como escalar JSON."""

        if value is None or isinstance(
            value,
            (str, bool, int),
        ):
            return True

        if isinstance(value, float):
            return isfinite(value)

        return False

    @staticmethod
    def _json_compatible_value(value: Any) -> Any:
        """
        Replica a normalização usada pelo ResponseGenerator.

        Isso permite validar valores como Decimal e datetime,
        que são serializados pelo gerador usando default=str.
        """

        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                default=str,
                allow_nan=False,
            )

        except (TypeError, ValueError, OverflowError):
            serialized = json.dumps(
                str(value),
                ensure_ascii=False,
            )

        return json.loads(serialized)

    @classmethod
    def _scalar_values_equal(
        cls,
        expected: Any,
        actual: Any,
    ) -> bool:
        """Compara escalares sem aceitar equivalência entre tipos diferentes."""

        if type(expected) is not type(actual):
            return False

        if isinstance(expected, float):
            if isnan(expected) or isnan(actual):
                return isnan(expected) and isnan(actual)

        return expected == actual

    # ==========================================================
    # VERIFICAÇÃO NUMÉRICA DAS AFIRMAÇÕES
    # ==========================================================

    @staticmethod
    def _normalize_number_token(
        token: str,
    ) -> Decimal | None:
        """Interpreta valores numéricos com convenção decimal brasileira."""

        normalized = token.strip().replace("%", "")

        if not normalized:
            return None

        sign = ""

        if normalized[0] in "+-":
            sign = normalized[0]
            normalized = normalized[1:]

        if "," in normalized:
            normalized = normalized.replace(".", "")
            normalized = normalized.replace(",", ".")

        elif "." in normalized:
            parts = normalized.split(".")

            if (
                len(parts) > 1
                and all(len(part) == 3 for part in parts[1:])
                and 1 <= len(parts[0]) <= 3
            ):
                normalized = "".join(parts)

        try:
            return Decimal(sign + normalized)

        except InvalidOperation:
            return None

    @classmethod
    def _extract_numeric_values(
        cls,
        text: str,
    ) -> set[Decimal]:
        """Extrai números de uma frase."""

        values = set()

        for match in cls.NUMBER_PATTERN.finditer(text):
            value = cls._normalize_number_token(
                match.group(),
            )

            if value is not None and value.is_finite():
                values.add(value)

        return values

    @classmethod
    def _extract_value_numbers(
        cls,
        value: Any,
    ) -> set[Decimal]:
        """Extrai números de uma célula referenciada."""

        if isinstance(value, bool) or value is None:
            return set()

        if isinstance(
            value,
            (int, float, Decimal),
        ):
            try:
                normalized = Decimal(str(value))

                if normalized.is_finite():
                    return {normalized}

            except (InvalidOperation, ValueError):
                return set()

            return set()

        if isinstance(value, str):
            return cls._extract_numeric_values(value)

        return set()

    # ==========================================================
    # NORMALIZAÇÃO DE TEXTO
    # ==========================================================

    @classmethod
    def _normalize_claim_text(
        cls,
        value: str,
    ) -> str:
        """
        Normaliza texto para comparar afirmações com a resposta.

        Ignora diferenças de acentos, caixa, pontuação e espaços.

        Os números são normalizados antes da remoção da pontuação,
        preservando a equivalência entre formatos brasileiros como
        1.200 e 1200, ou 1.200,50 e 1200,50.
        """

        normalized = unicodedata.normalize(
            "NFKD",
            value,
        )

        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )

        def normalize_numeric_match(
            match: re.Match[str],
        ) -> str:
            """Converte um número para uma representação canônica."""

            numeric_value = cls._normalize_number_token(
                match.group(),
            )

            if numeric_value is None:
                return match.group()

            canonical_value = format(
                numeric_value,
                "f",
            )

            if "." in canonical_value:
                canonical_value = canonical_value.rstrip("0").rstrip(".")

            return f" {canonical_value} "

        normalized = cls.NUMBER_PATTERN.sub(
            normalize_numeric_match,
            normalized,
        )

        tokens = re.findall(
            r"[a-z0-9]+(?:\.[0-9]+)?",
            normalized.casefold(),
        )

        return " ".join(tokens)

    # ==========================================================
    # VALIDAÇÃO AUXILIAR
    # ==========================================================

    @staticmethod
    def _optional_number(
        value: Any,
    ) -> float | None:
        """Normaliza metadados numéricos opcionais."""

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            return None

        value = float(value)

        if not isfinite(value) or value < 0:
            return None

        return value
