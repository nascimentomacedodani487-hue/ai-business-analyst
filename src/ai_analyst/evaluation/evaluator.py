"""Comparação determinística entre resultados esperados e reais."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from ai_analyst.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
)

_MISSING = object()


class ResultEvaluator:
    """Compara resultados do agente com referências confiáveis."""

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
        """Avalia um resultado e retorna seu relatório individual.

        A comparação considera:
        - Colunas obrigatórias.
        - Colunas inesperadas.
        - Quantidade de linhas.
        - Valores das células.
        - Tolerâncias numéricas.
        - Ordenação das linhas, quando configurada.

        Args:
            case: Cenário utilizado na avaliação.
            expected_rows: Resultado obtido pelo SQL de referência.
            actual_rows: Resultado produzido pelo agente.
            analysis_id: Identificador da análise original, se disponível.
            provider: Provedor de IA utilizado.
            model: Modelo utilizado.
            duration_ms: Duração da análise em milissegundos.
            cache_hit: Indica se o resultado veio do cache.

        Returns:
            EvaluationCaseResult com status, score e diferenças.
        """

        self._validate_rows(expected_rows, "expected_rows")
        self._validate_rows(actual_rows, "actual_rows")

        expected = [dict(row) for row in expected_rows]
        actual = [dict(row) for row in actual_rows]

        expected_columns = self._get_columns(expected)
        actual_columns = self._get_columns(actual)

        expected_schema = expected_columns | set(case.required_columns)

        differences: list[str] = []

        # 1. Validar o esquema de colunas.

        if actual:
            missing_columns = expected_schema - actual_columns

            for column in sorted(missing_columns):
                differences.append(
                    f"Coluna obrigatória ausente no resultado: '{column}'."
                )

            unexpected_columns = actual_columns - expected_schema

            if not case.allow_extra_columns:
                for column in sorted(unexpected_columns):
                    differences.append(f"Coluna inesperada no resultado: '{column}'.")

        # 2. Comparar as linhas.

        compared_columns = sorted(expected_schema)

        if case.sort_rows:
            matched_rows = self._compare_unordered_rows(
                expected,
                actual,
                compared_columns,
                case,
                differences,
            )
        else:
            matched_rows = self._compare_ordered_rows(
                expected,
                actual,
                compared_columns,
                case,
                differences,
            )

        # 3. Calcular o score.
        #
        # O score representa a proporção de linhas integralmente
        # correspondentes, com penalização adicional por erros
        # de esquema. Não representa precisão por célula.

        row_units = max(
            len(expected),
            len(actual),
            1,
        )

        schema_errors = sum(
            difference.startswith("Coluna ") for difference in differences
        )

        total_units = row_units + schema_errors

        score = matched_rows / total_units

        if not differences:
            score = 1.0

        score = round(score, 4)

        # 4. Determinar o status.

        status = "passed" if not differences else "failed"

        metadata = {
            "category": case.category,
            "comparison_mode": case.comparison_mode,
            "absolute_tolerance": case.absolute_tolerance,
            "relative_tolerance": case.relative_tolerance,
            "expected_row_count": len(expected),
            "actual_row_count": len(actual),
            "expected_columns": sorted(expected_schema),
            "actual_columns": sorted(actual_columns),
        }

        if not expected and not actual:
            metadata["schema_validation"] = "unavailable_empty_result"

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
    # VALIDAÇÃO E ESQUEMA
    # ==========================================================

    @staticmethod
    def _validate_rows(
        rows: list[dict[str, Any]],
        argument_name: str,
    ) -> None:
        """Valida o formato dos resultados recebidos."""

        if not isinstance(rows, list):
            raise TypeError(f"{argument_name} deve ser uma lista.")

        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"{argument_name}[{index}] deve ser um mapeamento.")

            if not all(isinstance(column, str) for column in row):
                raise TypeError(
                    f"Todas as colunas em {argument_name}[{index}] "
                    "devem possuir nomes do tipo string."
                )

    @staticmethod
    def _get_columns(
        rows: list[dict[str, Any]],
    ) -> set[str]:
        """Obtém todas as colunas presentes nos resultados."""

        columns: set[str] = set()

        for row in rows:
            columns.update(row.keys())

        return columns

    # ==========================================================
    # ASSOCIAÇÃO GLOBAL DE LINHAS
    # ==========================================================

    @staticmethod
    def _hungarian_rows_to_columns(
        cost_matrix: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Encontra a associação de menor custo entre linhas.

        Implementa o algoritmo Húngaro para matrizes em que
        a quantidade de linhas é menor ou igual à quantidade
        de colunas.

        Cada linha é associada a exatamente uma coluna.
        Uma coluna pode permanecer sem associação.

        Complexity: O(n² * m), onde n <= m.
        """

        row_count = len(cost_matrix)

        if row_count == 0:
            return []

        column_count = len(cost_matrix[0])

        if column_count == 0:
            return []

        if row_count > column_count:
            raise ValueError(
                "A matriz precisa possuir quantidade de linhas "
                "menor ou igual à quantidade de colunas."
            )

        if any(len(row) != column_count for row in cost_matrix):
            raise ValueError("A matriz de custos deve ser retangular.")

        # Potenciais duais das linhas e colunas.
        u = [0] * (row_count + 1)
        v = [0] * (column_count + 1)

        # p[j] indica qual linha está associada à coluna j.
        p = [0] * (column_count + 1)

        # way[j] registra o caminho utilizado na expansão.
        way = [0] * (column_count + 1)

        for row_index in range(1, row_count + 1):
            p[0] = row_index

            min_values = [float("inf")] * (column_count + 1)

            used = [False] * (column_count + 1)

            current_column = 0

            while True:
                used[current_column] = True

                current_row = p[current_column]

                delta = float("inf")
                next_column = 0

                for column_index in range(
                    1,
                    column_count + 1,
                ):
                    if used[column_index]:
                        continue

                    current_cost = (
                        cost_matrix[current_row - 1][column_index - 1]
                        - u[current_row]
                        - v[column_index]
                    )

                    if current_cost < min_values[column_index]:
                        min_values[column_index] = current_cost
                        way[column_index] = current_column

                    if min_values[column_index] < delta or (
                        min_values[column_index] == delta
                        and (next_column == 0 or column_index < next_column)
                    ):
                        delta = min_values[column_index]
                        next_column = column_index

                # Atualizar os potenciais duais.
                for column_index in range(column_count + 1):
                    if used[column_index]:
                        u[p[column_index]] += delta
                        v[column_index] -= delta
                    else:
                        min_values[column_index] -= delta

                current_column = next_column

                if p[current_column] == 0:
                    break

            # Reconstruir o caminho de associação.
            while True:
                previous_column = way[current_column]

                p[current_column] = p[previous_column]

                current_column = previous_column

                if current_column == 0:
                    break

        assignments = [
            (p[column_index] - 1, column_index - 1)
            for column_index in range(
                1,
                column_count + 1,
            )
            if p[column_index] != 0
        ]

        return sorted(assignments)

    @classmethod
    def _minimum_cost_assignment(
        cls,
        cost_matrix: list[list[int]],
    ) -> list[tuple[int, int]]:
        """Obtém a associação ótima para uma matriz retangular.

        Quando existem mais linhas esperadas do que linhas reais,
        a matriz é transposta e os índices são convertidos de volta.

        O resultado contém exatamente min(n, m) associações.
        """

        if not cost_matrix:
            return []

        row_count = len(cost_matrix)
        column_count = len(cost_matrix[0])

        if column_count == 0:
            return []

        if row_count <= column_count:
            return cls._hungarian_rows_to_columns(cost_matrix)

        transposed_matrix = [
            [cost_matrix[row_index][column_index] for row_index in range(row_count)]
            for column_index in range(column_count)
        ]

        transposed_assignments = cls._hungarian_rows_to_columns(transposed_matrix)

        return sorted(
            (
                expected_index,
                actual_index,
            )
            for actual_index, expected_index in transposed_assignments
        )

    def _row_mismatch_count(
        self,
        expected_row: dict[str, Any],
        actual_row: dict[str, Any],
        columns: list[str],
        case: EvaluationCase,
    ) -> int:
        """Conta as células diferentes entre duas linhas."""

        return sum(
            not self._values_equal(
                expected_row.get(column, _MISSING),
                actual_row.get(column, _MISSING),
                case,
            )
            for column in columns
        )

    def _compare_unordered_rows(
        self,
        expected: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        columns: list[str],
        case: EvaluationCase,
        differences: list[str],
    ) -> int:
        """Compara linhas sem exigir a mesma ordem.

        Utiliza associação global de custo mínimo para minimizar
        a quantidade total de células diferentes entre os pares.

        Linhas duplicadas são preservadas como registros distintos.
        Quando as quantidades diferem, as linhas excedentes são
        identificadas como ausentes ou inesperadas.
        """

        if not expected:
            for actual_index, actual_row in enumerate(actual):
                differences.append(f"Linha inesperada no resultado: {actual_row!r}.")

            return 0

        if not actual:
            for expected_row in expected:
                differences.append(
                    f"Linha esperada ausente no resultado: {expected_row!r}."
                )

            return 0

        # 1. Construir a matriz de custos.
        #
        # Cada célula representa a quantidade de diferenças
        # entre uma linha esperada e uma linha real.

        cost_matrix = [
            [
                self._row_mismatch_count(
                    expected_row,
                    actual_row,
                    columns,
                    case,
                )
                for actual_row in actual
            ]
            for expected_row in expected
        ]

        # 2. Encontrar a associação global de menor custo.

        assignments = self._minimum_cost_assignment(cost_matrix)

        assigned_expected: set[int] = set()
        assigned_actual: set[int] = set()

        matched_rows = 0

        # 3. Descrever as diferenças de cada associação.

        for expected_index, actual_index in assignments:
            assigned_expected.add(expected_index)
            assigned_actual.add(actual_index)

            expected_row = expected[expected_index]
            actual_row = actual[actual_index]

            if self._rows_equal(
                expected_row,
                actual_row,
                columns,
                case,
            ):
                matched_rows += 1
                continue

            differences.extend(
                self._row_differences(
                    expected_row,
                    actual_row,
                    columns,
                    expected_index,
                    actual_index,
                    case,
                )
            )

        # 4. Identificar linhas esperadas sem associação.

        for expected_index, expected_row in enumerate(expected):
            if expected_index not in assigned_expected:
                differences.append(
                    f"Linha esperada ausente no resultado: {expected_row!r}."
                )

        # 5. Identificar linhas reais sem associação.

        for actual_index, actual_row in enumerate(actual):
            if actual_index not in assigned_actual:
                differences.append(f"Linha inesperada no resultado: {actual_row!r}.")

        return matched_rows

    # ==========================================================
    # COMPARAÇÃO ORDENADA
    # ==========================================================

    def _compare_ordered_rows(
        self,
        expected: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        columns: list[str],
        case: EvaluationCase,
        differences: list[str],
    ) -> int:
        """Compara linhas preservando a posição de cada registro."""

        matched_rows = 0

        common_length = min(
            len(expected),
            len(actual),
        )

        for index in range(common_length):
            expected_row = expected[index]
            actual_row = actual[index]

            if self._rows_equal(
                expected_row,
                actual_row,
                columns,
                case,
            ):
                matched_rows += 1
                continue

            differences.extend(
                self._row_differences(
                    expected_row,
                    actual_row,
                    columns,
                    index,
                    index,
                    case,
                )
            )

        for index in range(common_length, len(expected)):
            differences.append(
                f"Linha {index + 1} esperada ausente: {expected[index]!r}."
            )

        for index in range(common_length, len(actual)):
            differences.append(
                f"Linha inesperada na posição {index + 1}: {actual[index]!r}."
            )

        return matched_rows

    # ==========================================================
    # DIAGNÓSTICO DAS DIFERENÇAS
    # ==========================================================

    def _row_differences(
        self,
        expected_row: dict[str, Any],
        actual_row: dict[str, Any],
        columns: list[str],
        expected_index: int,
        actual_index: int | None,
        case: EvaluationCase,
    ) -> list[str]:
        """Descreve as diferenças encontradas em uma linha."""

        row_label = f"Linha esperada {expected_index + 1}"

        if actual_index is not None:
            row_label += f" / linha real {actual_index + 1}"

        row_differences: list[str] = []

        for column in columns:
            expected_value = expected_row.get(
                column,
                _MISSING,
            )

            actual_value = actual_row.get(
                column,
                _MISSING,
            )

            if not self._values_equal(
                expected_value,
                actual_value,
                case,
            ):
                expected_display = (
                    "<coluna ausente>"
                    if expected_value is _MISSING
                    else repr(expected_value)
                )

                actual_display = (
                    "<coluna ausente>"
                    if actual_value is _MISSING
                    else repr(actual_value)
                )

                row_differences.append(
                    f"{row_label}, coluna '{column}': "
                    f"esperado {expected_display}, "
                    f"recebido {actual_display}."
                )

        return row_differences

    def _rows_equal(
        self,
        expected_row: dict[str, Any],
        actual_row: dict[str, Any],
        columns: list[str],
        case: EvaluationCase,
    ) -> bool:
        """Verifica se duas linhas possuem os mesmos valores."""

        return all(
            self._values_equal(
                expected_row.get(column, _MISSING),
                actual_row.get(column, _MISSING),
                case,
            )
            for column in columns
        )

    # ==========================================================
    # COMPARAÇÃO DE VALORES
    # ==========================================================

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """Identifica valores numéricos sem tratar bool como número."""

        return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)

    @classmethod
    def _values_equal(
        cls,
        expected: Any,
        actual: Any,
        case: EvaluationCase,
    ) -> bool:
        """Compara dois valores respeitando a política do cenário."""

        if expected is _MISSING or actual is _MISSING:
            return expected is actual

        if expected is None or actual is None:
            return expected is actual

        expected_numeric = cls._is_numeric(expected)
        actual_numeric = cls._is_numeric(actual)

        if expected_numeric != actual_numeric:
            return False

        if not expected_numeric:
            return expected == actual

        try:
            expected_decimal = Decimal(str(expected))
            actual_decimal = Decimal(str(actual))

        except (InvalidOperation, ValueError):
            return False

        # NaN deve ser tratado antes da comparação exata,
        # pois NaN != NaN na semântica numérica tradicional.
        if expected_decimal.is_nan() or actual_decimal.is_nan():
            return expected_decimal.is_nan() and actual_decimal.is_nan()

        # Comparação exata, incluindo valores infinitos.
        if case.comparison_mode == "exact":
            return expected_decimal == actual_decimal

        # Infinitos não devem participar do cálculo de tolerância.
        if not expected_decimal.is_finite() or not actual_decimal.is_finite():
            return expected_decimal == actual_decimal

        difference = abs(expected_decimal - actual_decimal)

        absolute_tolerance = Decimal(str(case.absolute_tolerance))

        relative_tolerance = Decimal(str(case.relative_tolerance))

        allowed_difference = max(
            absolute_tolerance,
            relative_tolerance * abs(expected_decimal),
        )

        return difference <= allowed_difference
