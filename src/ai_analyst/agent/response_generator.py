"""
Geração de respostas analíticas em linguagem natural.

Responsabilidades:
- Interpretar resultados de consultas SQL.
- Sintetizar múltiplas consultas independentes.
- Preservar os métodos legados que retornam texto.
- Gerar respostas estruturadas com referências às evidências.
- Validar referências a consultas, linhas, colunas e valores.
- Limitar o contexto enviado aos provedores de IA.
- Identificar truncamentos de linhas, colunas e células.
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from typing import Any

from ai_analyst.llm.base import LLMProvider, LLMResponseError

logger = logging.getLogger(__name__)

MAX_SYNTHESIS_ROWS = 30
MAX_SYNTHESIS_COLUMNS = 20
MAX_CELL_LENGTH = 300
MAX_SYNTHESIS_PAYLOAD_CHARS = 24000

MAX_SYNTHESIS_TEXT_LENGTH = 1000
MAX_SYNTHESIS_QUESTION_LENGTH = 5000
MAX_SYNTHESIS_SQL_LENGTH = 4000

MAX_STRUCTURED_CLAIMS = 50
MAX_EVIDENCE_PER_CLAIM = 20
MAX_CLAIM_LENGTH = 2000
MAX_ANSWER_LENGTH = 10000

STRUCTURED_RESPONSE_MAX_TOKENS = 1200


class ResponseGenerator:
    """Transforma resultados SQL em explicações de negócio."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        gemini: LLMProvider | None = None,
    ):
        if provider is not None and gemini is not None:
            if provider is not gemini:
                raise ValueError("Informe apenas um provedor de IA.")

        self.provider = provider or gemini

        if self.provider is None:
            raise ValueError("Um provedor de IA deve ser informado.")

        # Compatibilidade com consumidores legados.
        self.gemini = self.provider

    # ==========================================================
    # GERAÇÃO LEGADA
    # ==========================================================

    def generate(
        self,
        question: str,
        sql: str,
        results: list[dict],
    ) -> str:
        """
        Interpreta resultados de uma consulta SQL.

        Mantém o contrato legado: retorna somente uma string.
        """

        if not results:
            return "Não foram encontrados dados para responder à pergunta."

        payload = self._build_single_payload(
            question=question,
            results=results,
        )

        serialized_results = self._serialize_payload(payload)

        prompt = f"""
Pergunta do usuário:
{
            self._limit_text(
                question,
                MAX_SYNTHESIS_QUESTION_LENGTH,
            )
        }

SQL executado:
{
            self._limit_text(
                sql,
                MAX_SYNTHESIS_SQL_LENGTH,
            )
        }

Resultados do banco de dados:
{serialized_results}

Instruções:
- Responda em português brasileiro.
- Seja objetivo e profissional.
- Utilize exclusivamente os resultados apresentados.
- Não invente números, causas ou conclusões.
- Destaque os principais números relevantes à pergunta.
- Quando houver comparações, identifique diferenças importantes.
- Se os dados não permitirem uma conclusão, informe isso.
- Não apresente o SQL na resposta.
- Evite introduções genéricas e repetições.
- Prefira uma resposta curta e diretamente relacionada à pergunta.
- Observe os metadados de truncamento dos resultados.
- Quando os resultados estiverem truncados, restrinja suas
  conclusões aos registros e valores efetivamente apresentados.
"""

        return self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self._system_prompt(),
            max_output_tokens=512,
        )

    def generate_multi(
        self,
        question: str,
        queries: list[dict],
    ) -> str:
        """
        Sintetiza os resultados de múltiplas consultas independentes.

        Cada item deve conter:
        - objective
        - question
        - status
        - results
        - total_results
        - error (opcional)

        Mantém o contrato legado: retorna somente uma string.
        """

        self._validate_multi_input(question, queries)

        payload = self._build_multi_payload(queries)

        successful_results = [
            item for item in payload if item["status"] == "success" and item["results"]
        ]

        if not successful_results:
            return self._build_no_results_answer(payload)

        serialized_payload = self._serialize_payload(payload)

        prompt = f"""
PERGUNTA ORIGINAL DO USUÁRIO:
{
            self._limit_text(
                question,
                MAX_SYNTHESIS_QUESTION_LENGTH,
            )
        }

RESULTADOS DAS CONSULTAS INDEPENDENTES:
{serialized_payload}

INSTRUÇÕES DE SÍNTESE:

1. Responda em português brasileiro.
2. Considere cada objetivo separadamente.
3. Utilize exclusivamente os dados apresentados.
4. Não invente números, causas, tendências ou conclusões.
5. Não misture métricas de consultas diferentes.
6. Faça comparações somente quando os dados permitirem.
7. Identifique claramente resultados parciais.
8. Não trate consultas com falha como resultados válidos.
9. Cada consulta informa total_results, provided_results,
   omitted_results e results_truncated.
10. Quando results_truncated for true, considere os registros
    enviados como uma amostra parcial dos resultados.
11. Nunca afirme que meses, categorias ou status ausentes da
    amostra não existem na base.
12. Não apresente totais ou tendências do conjunto completo
    quando os resultados estiverem truncados.
13. Observe também columns_truncated e cells_truncated.
14. Não utilize valores de células que tenham sido truncadas
    para construir conclusões.
15. Se os resultados forem insuficientes, informe isso.
16. Não apresente SQL.
17. Evite repetições e introduções genéricas.

Produza uma resposta integrada, organizada e objetiva,
destacando os principais achados de cada objetivo.
"""

        return self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self._system_prompt(multi=True),
            max_output_tokens=900,
        )

    # ==========================================================
    # GERAÇÃO ESTRUTURADA COM EVIDÊNCIAS
    # ==========================================================

    def generate_with_evidence(
        self,
        question: str,
        sql: str,
        results: list[dict],
    ) -> dict[str, Any]:
        """
        Gera uma resposta simples acompanhada de evidências estruturadas.

        Contrato de retorno:

        {
            "answer": "...",
            "answer_evidence": [
                {
                    "claim": "...",
                    "evidence": [
                        {
                            "query_id": 1,
                            "row_index": 0,
                            "column": "total",
                            "value": 100
                        }
                    ]
                }
            ]
        }

        As referências são verificadas contra os resultados
        efetivamente enviados ao provedor.
        """

        if not results:
            return {
                "answer": ("Não foram encontrados dados para responder à pergunta."),
                "answer_evidence": [],
            }

        payload = self._build_single_payload(
            question=question,
            results=results,
        )

        serialized_payload = self._serialize_payload(payload)

        prompt = f"""
PERGUNTA ORIGINAL DO USUÁRIO:
{
            self._limit_text(
                question,
                MAX_SYNTHESIS_QUESTION_LENGTH,
            )
        }

SQL EXECUTADO:
{
            self._limit_text(
                sql,
                MAX_SYNTHESIS_SQL_LENGTH,
            )
        }

RESULTADOS DISPONÍVEIS:
{serialized_payload}

Sua tarefa é produzir uma resposta analítica fundamentada
exclusivamente nos resultados apresentados.

REGRAS OBRIGATÓRIAS:

1. Responda em português brasileiro.
2. Não invente números, causas, tendências ou conclusões.
3. Não utilize conhecimentos externos para completar informações.
4. Toda afirmação factual ou quantitativa da resposta deve
   aparecer individualmente em claims.
5. Cada afirmação deve possuir pelo menos uma evidência.
6. Uma evidência deve apontar para um registro existente.
7. Utilize somente consultas, linhas e colunas disponíveis.
8. O row_index começa em zero e representa a posição da linha
   dentro de results, exatamente como apresentada.
9. O query_id deve ser exatamente o identificador da consulta.
10. O valor deve corresponder exatamente ao conteúdo da célula.
11. Não utilize valores de células indicadas em truncated_cells.
12. Não cite células, colunas ou registros ausentes.
13. Quando os resultados estiverem truncados, não generalize
    conclusões para a base inteira.
14. Se os dados forem insuficientes, explique a limitação.
15. Não apresente o SQL na resposta.

FORMATO DE SAÍDA:

Retorne exclusivamente um objeto JSON válido.
Não utilize Markdown, cercas de código ou explicações externas.

A estrutura obrigatória é:

{{
    "answer": "Resposta em linguagem natural.",
    "claims": [
        {{
            "claim": "Afirmação fundamentada nos dados.",
            "evidence": [
                {{
                    "query_id": 1,
                    "row_index": 0,
                    "column": "nome_da_coluna",
                    "value": 100
                }}
            ]
        }}
    ]
}}

O campo claims deve conter todas as afirmações factuais
apresentadas na resposta.

Se não houver afirmações factuais sustentáveis,
retorne claims como uma lista vazia.

Não invente referências ou valores para preencher o formato.
"""

        raw_response = self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self._structured_system_prompt(),
            max_output_tokens=STRUCTURED_RESPONSE_MAX_TOKENS,
        )

        return self._parse_and_validate_structured_response(
            raw_response,
            payload,
        )

    def generate_multi_with_evidence(
        self,
        question: str,
        queries: list[dict],
    ) -> dict[str, Any]:
        """
        Sintetiza múltiplas consultas com afirmações e evidências.

        Preserva a identificação individual de cada consulta,
        inclusive quando existem falhas ou sucesso parcial.
        """

        self._validate_multi_input(question, queries)

        payload = self._build_multi_payload(queries)

        successful_results = [
            item for item in payload if item["status"] == "success" and item["results"]
        ]

        if not successful_results:
            return {
                "answer": self._build_no_results_answer(payload),
                "answer_evidence": [],
            }

        serialized_payload = self._serialize_payload(payload)

        prompt = f"""
PERGUNTA ORIGINAL DO USUÁRIO:
{
            self._limit_text(
                question,
                MAX_SYNTHESIS_QUESTION_LENGTH,
            )
        }

RESULTADOS DAS CONSULTAS INDEPENDENTES:
{serialized_payload}

Sua tarefa é produzir uma síntese analítica fundamentada
exclusivamente nos resultados apresentados.

REGRAS OBRIGATÓRIAS:

1. Responda em português brasileiro.
2. Considere cada objetivo separadamente.
3. Não invente números, causas, tendências ou conclusões.
4. Não misture métricas de consultas diferentes.
5. Não utilize consultas com status failure como evidência.
6. Identifique claramente limitações e resultados parciais.
7. Toda afirmação factual ou quantitativa da resposta deve
   aparecer individualmente em claims.
8. Cada afirmação deve possuir pelo menos uma evidência.
9. Uma evidência deve apontar para uma consulta concluída.
10. Utilize exatamente o query_id da consulta original.
11. O row_index começa em zero e representa a posição da linha
    dentro dos resultados apresentados para aquela consulta.
12. A coluna e o valor devem corresponder exatamente ao registro.
13. Não utilize valores de células indicadas em truncated_cells.
14. Não cite registros ou colunas ausentes.
15. Quando results_truncated for true, restrinja as conclusões
    aos registros efetivamente apresentados.
16. Não apresente totais ou tendências do conjunto completo
    quando os resultados estiverem truncados.
17. Não invente informações para completar consultas com falha.
18. Se os dados forem insuficientes, explique a limitação.
19. Não apresente SQL.

FORMATO DE SAÍDA:

Retorne exclusivamente um objeto JSON válido.
Não utilize Markdown, cercas de código ou explicações externas.

A estrutura obrigatória é:

{{
    "answer": "Resposta consolidada em linguagem natural.",
    "claims": [
        {{
            "claim": "Afirmação fundamentada nos dados.",
            "evidence": [
                {{
                    "query_id": 1,
                    "row_index": 0,
                    "column": "nome_da_coluna",
                    "value": 100
                }}
            ]
        }}
    ]
}}

O campo claims deve conter todas as afirmações factuais
apresentadas na resposta.

Se não houver afirmações factuais sustentáveis,
retorne claims como uma lista vazia.

Uma mesma afirmação pode utilizar mais de uma evidência.
Cada referência deve apontar para uma célula existente.

Não invente referências ou valores para preencher o formato.
"""

        raw_response = self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self._structured_system_prompt(multi=True),
            max_output_tokens=STRUCTURED_RESPONSE_MAX_TOKENS,
        )

        return self._parse_and_validate_structured_response(
            raw_response,
            payload,
        )

    # ==========================================================
    # PREPARAÇÃO E VALIDAÇÃO DOS DADOS DE ENTRADA
    # ==========================================================

    @staticmethod
    def _validate_multi_input(
        question: str,
        queries: list[dict],
    ) -> None:
        """Valida os dados recebidos para síntese de múltiplas consultas."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("A pergunta original não pode estar vazia.")

        if not isinstance(queries, list) or not queries:
            raise ValueError("É necessário informar pelo menos uma consulta.")

    @classmethod
    def _build_single_payload(
        cls,
        question: str,
        results: list[dict],
    ) -> list[dict]:
        """Prepara os resultados de uma análise simples."""

        if not isinstance(results, list):
            raise ValueError("Os resultados devem ser uma lista.")

        if not all(isinstance(row, dict) for row in results):
            raise ValueError("Os registros devem ser objetos.")

        limited_rows, truncation = cls._limit_rows_with_metadata(results)

        payload = [
            {
                "id": 1,
                "objective": cls._limit_text(
                    question,
                    MAX_SYNTHESIS_TEXT_LENGTH,
                ),
                "question": cls._limit_text(
                    question,
                    MAX_SYNTHESIS_TEXT_LENGTH,
                ),
                "status": "success",
                "total_results": len(results),
                "results": limited_rows,
                "error": None,
                **truncation,
            }
        ]

        return payload

    @classmethod
    def _build_multi_payload(
        cls,
        queries: list[dict],
    ) -> list[dict]:
        """Valida e prepara as consultas sem modificar os objetos originais."""

        payload = []
        seen_ids = set()

        for index, query in enumerate(queries, start=1):
            if not isinstance(query, dict):
                raise ValueError(f"A consulta {index} deve ser um objeto.")

            status = query.get("status")

            if status not in {"success", "failure"}:
                raise ValueError(f"Status inválido na consulta {index}.")

            results = query.get("results", [])

            if not isinstance(results, list):
                raise ValueError(
                    f"Os resultados da consulta {index} devem ser uma lista."
                )

            if not all(isinstance(row, dict) for row in results):
                raise ValueError(f"Os registros da consulta {index} devem ser objetos.")

            total_results = query.get(
                "total_results",
                len(results),
            )

            if (
                isinstance(total_results, bool)
                or not isinstance(total_results, int)
                or total_results < len(results)
            ):
                raise ValueError(f"total_results inválido na consulta {index}.")

            query_id = query.get("id", index)

            if isinstance(query_id, bool) or not isinstance(query_id, (str, int)):
                raise ValueError(
                    f"O identificador da consulta {index} deve ser texto ou inteiro."
                )

            if isinstance(query_id, str) and not query_id.strip():
                raise ValueError(
                    f"O identificador da consulta {index} não pode estar vazio."
                )

            normalized_id = (type(query_id).__name__, query_id)

            if normalized_id in seen_ids:
                raise ValueError(f"Identificador duplicado na consulta {index}.")

            seen_ids.add(normalized_id)

            limited_rows, truncation = cls._limit_rows_with_metadata(
                results,
            )

            payload.append(
                {
                    "id": query_id,
                    "objective": cls._limit_text(
                        str(query.get("objective") or ""),
                        MAX_SYNTHESIS_TEXT_LENGTH,
                    ),
                    "question": cls._limit_text(
                        str(query.get("question") or ""),
                        MAX_SYNTHESIS_TEXT_LENGTH,
                    ),
                    "status": status,
                    "total_results": total_results,
                    "results": limited_rows,
                    "error": (
                        cls._limit_text(
                            str(query["error"]),
                            MAX_SYNTHESIS_TEXT_LENGTH,
                        )
                        if query.get("error")
                        else None
                    ),
                    **truncation,
                }
            )

        return payload

    @staticmethod
    def _limit_text(
        value: str,
        maximum: int,
    ) -> str:
        """Limita um texto sem modificar a string original."""

        if len(value) <= maximum:
            return value

        return value[:maximum]

    # ==========================================================
    # LIMITES E METADADOS DO CONTEXTO
    # ==========================================================

    @staticmethod
    def _limit_rows(results: list[dict]) -> list[dict]:
        """Compatibilidade com o helper anterior."""

        limited, _ = ResponseGenerator._limit_rows_with_metadata(results)

        return limited

    @staticmethod
    def _limit_rows_with_metadata(
        results: list[dict],
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Limita linhas, colunas e células.

        Registra quais células foram cortadas para impedir que
        sejam utilizadas como evidências estruturadas.
        """

        limited = []

        columns_truncated = False
        omitted_columns = 0

        cells_truncated = False
        truncated_cells = []

        for row_index, row in enumerate(results[:MAX_SYNTHESIS_ROWS]):
            limited_row = {}
            columns = list(row.items())

            if len(columns) > MAX_SYNTHESIS_COLUMNS:
                columns_truncated = True
                omitted_columns += len(columns) - MAX_SYNTHESIS_COLUMNS

            for column, value in columns[:MAX_SYNTHESIS_COLUMNS]:
                column_name = str(column)

                if isinstance(value, str) and len(value) > MAX_CELL_LENGTH:
                    cells_truncated = True

                    truncated_cells.append(
                        {
                            "row_index": row_index,
                            "column": column_name,
                        }
                    )

                    value = value[:MAX_CELL_LENGTH]

                limited_row[column_name] = value

            limited.append(limited_row)

        metadata = {
            "columns_truncated": columns_truncated,
            "omitted_columns": omitted_columns,
            "cells_truncated": cells_truncated,
            "truncated_cells": truncated_cells,
        }

        return limited, metadata

    @classmethod
    def _serialize_payload(
        cls,
        payload: list[dict],
    ) -> str:
        """
        Serializa o contexto respeitando o limite de caracteres.

        A operação trabalha sobre uma cópia, preservando os dados
        utilizados pelo agente e pelo histórico.
        """

        working_payload = deepcopy(payload)

        for item in working_payload:
            cls._refresh_payload_metadata(item)

        serialized = cls._dump_payload(working_payload)

        while len(serialized) > MAX_SYNTHESIS_PAYLOAD_CHARS:
            candidates = [item for item in working_payload if item["results"]]

            if not candidates:
                break

            largest = max(
                candidates,
                key=lambda item: len(item["results"]),
            )

            largest["results"].pop()

            cls._refresh_payload_metadata(largest)

            serialized = cls._dump_payload(working_payload)

        # Reduz textos descritivos caso o tamanho excedente
        # venha de perguntas, objetivos ou mensagens de erro.
        if len(serialized) > MAX_SYNTHESIS_PAYLOAD_CHARS:
            for item in working_payload:
                for field in ("objective", "question", "error"):
                    value = item.get(field)

                    if isinstance(value, str):
                        item[field] = value[:120]

                if isinstance(item.get("id"), str):
                    item["id"] = item["id"][:100]

            serialized = cls._dump_payload(working_payload)

        if len(serialized) > MAX_SYNTHESIS_PAYLOAD_CHARS:
            raise ValueError(
                "O contexto das consultas excedeu o limite de síntese "
                "mesmo após a redução dos resultados."
            )

        return serialized

    @staticmethod
    def _refresh_payload_metadata(
        item: dict,
    ) -> None:
        """Atualiza os indicadores de truncamento de uma consulta."""

        provided_results = len(item["results"])

        omitted_results = max(
            item["total_results"] - provided_results,
            0,
        )

        item["provided_results"] = provided_results
        item["omitted_results"] = omitted_results
        item["rows_truncated"] = omitted_results > 0

        item["truncated_cells"] = [
            cell
            for cell in item.get("truncated_cells", [])
            if cell["row_index"] < provided_results
        ]

        item["results_truncated"] = any(
            (
                item["rows_truncated"],
                item.get("columns_truncated", False),
                item.get("cells_truncated", False),
            )
        )

    @staticmethod
    def _dump_payload(payload: list[dict]) -> str:
        """Serializa o payload de forma consistente."""

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

    # ==========================================================
    # VALIDAÇÃO DAS RESPOSTAS ESTRUTURADAS
    # ==========================================================

    def _parse_and_validate_structured_response(
        self,
        raw_response: str,
        payload: list[dict],
    ) -> dict[str, Any]:
        """
        Converte a resposta JSON do provedor e valida suas evidências.

        Respostas malformadas ou referências sem correspondência
        nos resultados originais são rejeitadas.
        """

        parsed = self._parse_structured_json(raw_response)

        if not isinstance(parsed, dict):
            raise LLMResponseError(
                "A resposta estruturada do provedor não é um objeto JSON."
            )

        if set(parsed) != {"answer", "claims"}:
            raise LLMResponseError(
                "A resposta estruturada não contém os campos obrigatórios."
            )

        answer = parsed.get("answer")
        claims = parsed.get("claims")

        if not isinstance(answer, str) or not answer.strip():
            raise LLMResponseError(
                "A resposta estruturada não contém uma resposta válida."
            )

        answer = answer.strip()

        if len(answer) > MAX_ANSWER_LENGTH:
            raise LLMResponseError("A resposta estruturada excedeu o limite permitido.")

        if not isinstance(claims, list):
            raise LLMResponseError("O campo claims deve ser uma lista.")

        if len(claims) > MAX_STRUCTURED_CLAIMS:
            raise LLMResponseError(
                "A resposta contém afirmações acima do limite permitido."
            )

        validated_claims = self._validate_claims(
            claims,
            payload,
        )

        return {
            "answer": answer,
            "answer_evidence": validated_claims,
        }

    @staticmethod
    def _parse_structured_json(raw_response: str) -> Any:
        """Interpreta JSON puro ou uma resposta envolvida em Markdown."""

        if not isinstance(raw_response, str) or not raw_response.strip():
            raise LLMResponseError(
                "O provedor retornou uma resposta estruturada vazia."
            )

        candidate = raw_response.strip()

        if candidate.startswith("```"):
            lines = candidate.splitlines()

            if (
                len(lines) >= 3
                and lines[0].strip().startswith("```")
                and lines[-1].strip() == "```"
            ):
                candidate = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(candidate)

        except json.JSONDecodeError as exc:
            logger.warning(
                "O provedor retornou JSON inválido para a resposta estruturada.",
                extra={
                    "event": "structured_response_invalid_json",
                },
            )

            raise LLMResponseError(
                "O provedor retornou uma resposta estruturada em JSON inválido."
            ) from exc

    @classmethod
    def _validate_claims(
        cls,
        claims: list,
        payload: list[dict],
    ) -> list[dict[str, Any]]:
        """
        Valida cada afirmação e suas referências.

        Uma evidência somente é aceita quando:
        - A consulta existe.
        - A consulta foi concluída.
        - A linha existe no contexto enviado.
        - A coluna existe na linha.
        - A célula não foi truncada.
        - O valor informado corresponde exatamente ao valor da célula.
        """

        query_map = {}

        for query in payload:
            query_id = query["id"]

            query_map[(type(query_id).__name__, query_id)] = query

        validated_claims = []

        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                raise LLMResponseError(
                    f"A afirmação {claim_index + 1} possui formato inválido."
                )

            if set(claim) != {"claim", "evidence"}:
                raise LLMResponseError(
                    f"A afirmação {claim_index + 1} possui campos inválidos."
                )

            claim_text = claim.get("claim")
            evidence = claim.get("evidence")

            if (
                not isinstance(claim_text, str)
                or not claim_text.strip()
                or len(claim_text) > MAX_CLAIM_LENGTH
            ):
                raise LLMResponseError(
                    f"A afirmação {claim_index + 1} possui texto inválido."
                )

            if not isinstance(evidence, list) or not evidence:
                raise LLMResponseError(
                    f"A afirmação {claim_index + 1} não possui evidências."
                )

            if len(evidence) > MAX_EVIDENCE_PER_CLAIM:
                raise LLMResponseError(
                    f"A afirmação {claim_index + 1} excedeu o limite de evidências."
                )

            validated_evidence = []

            for evidence_index, reference in enumerate(evidence):
                validated_evidence.append(
                    cls._validate_evidence_reference(
                        reference,
                        query_map,
                        claim_index=claim_index,
                        evidence_index=evidence_index,
                    )
                )

            validated_claims.append(
                {
                    "claim": claim_text.strip(),
                    "evidence": validated_evidence,
                }
            )

        return validated_claims

    @classmethod
    def _validate_evidence_reference(
        cls,
        reference: Any,
        query_map: dict,
        *,
        claim_index: int,
        evidence_index: int,
    ) -> dict[str, Any]:
        """Valida individualmente uma referência de evidência."""

        location = f"afirmação {claim_index + 1}, evidência {evidence_index + 1}"

        if not isinstance(reference, dict):
            raise LLMResponseError(f"A referência da {location} deve ser um objeto.")

        required_fields = {
            "query_id",
            "row_index",
            "column",
            "value",
        }

        if set(reference) != required_fields:
            raise LLMResponseError(
                f"A referência da {location} possui campos inválidos."
            )

        query_id = reference["query_id"]
        row_index = reference["row_index"]
        column = reference["column"]
        cited_value = reference["value"]

        if isinstance(query_id, bool) or not isinstance(query_id, (str, int)):
            raise LLMResponseError(
                f"O identificador da consulta da {location} é inválido."
            )

        if isinstance(query_id, str) and not query_id.strip():
            raise LLMResponseError(
                f"O identificador da consulta da {location} está vazio."
            )

        if isinstance(row_index, bool) or not isinstance(row_index, int):
            raise LLMResponseError(f"O índice da linha da {location} é inválido.")

        if row_index < 0:
            raise LLMResponseError(
                f"O índice da linha da {location} não pode ser negativo."
            )

        if not isinstance(column, str) or not column.strip():
            raise LLMResponseError(f"A coluna da {location} é inválida.")

        if not cls._is_json_scalar(cited_value):
            raise LLMResponseError(
                f"O valor da referência da {location} não é um valor escalar válido."
            )

        query = query_map.get(
            (type(query_id).__name__, query_id),
        )

        if query is None:
            raise LLMResponseError(f"A consulta referenciada na {location} não existe.")

        if query["status"] != "success":
            raise LLMResponseError(
                f"A consulta referenciada na {location} não foi concluída."
            )

        rows = query["results"]

        if row_index >= len(rows):
            raise LLMResponseError(f"A linha referenciada na {location} não existe.")

        row = rows[row_index]

        if column not in row:
            raise LLMResponseError(f"A coluna referenciada na {location} não existe.")

        truncated_cells = query.get("truncated_cells", [])

        if any(
            cell["row_index"] == row_index and cell["column"] == column
            for cell in truncated_cells
        ):
            raise LLMResponseError(f"A célula referenciada na {location} foi truncada.")

        actual_value = row[column]

        if isinstance(actual_value, (dict, list, tuple, set)):
            raise LLMResponseError(
                f"A célula referenciada na {location} não contém um valor escalar."
            )

        normalized_actual = cls._json_compatible_value(actual_value)

        if (
            type(normalized_actual) is not type(cited_value)
            or normalized_actual != cited_value
        ):
            raise LLMResponseError(
                f"O valor informado na {location} não corresponde "
                "ao resultado original."
            )

        return {
            "query_id": query_id,
            "row_index": row_index,
            "column": column,
            "value": cited_value,
        }

    @staticmethod
    def _is_json_scalar(value: Any) -> bool:
        """Verifica se o valor é um escalar JSON válido."""

        if value is None or isinstance(value, (str, bool, int)):
            return True

        if isinstance(value, float):
            return math.isfinite(value)

        return False

    @staticmethod
    def _json_compatible_value(value: Any) -> Any:
        """
        Converte valores de banco para a representação JSON enviada ao LLM.

        Valores como Decimal e datetime são convertidos para string,
        conforme o default=str utilizado na serialização do contexto.
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

    # ==========================================================
    # RESPOSTAS SEM RESULTADOS
    # ==========================================================

    @staticmethod
    def _build_no_results_answer(
        payload: list[dict],
    ) -> str:
        """Monta uma resposta determinística sem chamar a IA."""

        failures = [item for item in payload if item["status"] == "failure"]

        if failures:
            completed = len(payload) - len(failures)

            if completed:
                return (
                    f"{completed} consulta(s) foram concluídas, "
                    "mas não retornaram registros. "
                    f"{len(failures)} consulta(s) falharam. "
                    "Não foi possível produzir uma síntese "
                    "com os dados disponíveis."
                )

            return (
                "Não foi possível concluir as consultas. "
                "Nenhum resultado válido foi disponibilizado "
                "para análise."
            )

        return (
            "As consultas foram executadas, mas não foram "
            "encontrados registros para responder à pergunta."
        )

    # ==========================================================
    # PROMPTS DE SISTEMA
    # ==========================================================

    @staticmethod
    def _system_prompt(multi: bool = False) -> str:
        """Retorna as instruções legadas do analista."""

        role = (
            "Você é um analista de negócios especializado "
            "em síntese de múltiplas análises independentes. "
            if multi
            else "Você é um analista de negócios especializado "
            "em interpretação de dados. "
        )

        return (
            role + "Sua função é transformar resultados SQL em "
            "respostas claras, precisas e objetivas. "
            "Nunca invente informações."
        )

    @staticmethod
    def _structured_system_prompt(multi: bool = False) -> str:
        """Retorna as instruções para geração estruturada."""

        role = (
            "Você é um analista de negócios especializado "
            "em síntese fundamentada de múltiplas consultas SQL. "
            if multi
            else "Você é um analista de negócios especializado "
            "em interpretação fundamentada de resultados SQL. "
        )

        return (
            role + "Sua responsabilidade é produzir respostas em JSON válido, "
            "identificando individualmente as afirmações factuais e "
            "suas referências exatas aos resultados apresentados. "
            "Não invente valores, referências, causas ou conclusões. "
            "Não utilize informações externas. "
            "Quando não houver evidência suficiente, informe a limitação."
        )
