"""
Geração de respostas analíticas em linguagem natural.
"""

import json
import logging

from ai_analyst.llm.base import LLMProvider

logger = logging.getLogger(__name__)

MAX_SYNTHESIS_ROWS = 30
MAX_SYNTHESIS_COLUMNS = 20
MAX_CELL_LENGTH = 300
MAX_SYNTHESIS_PAYLOAD_CHARS = 24000


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

        self.gemini = self.provider

    def generate(
        self,
        question: str,
        sql: str,
        results: list[dict],
    ) -> str:
        """Interpreta resultados de uma consulta SQL."""

        if not results:
            return "Não foram encontrados dados para responder à pergunta."

        serialized_results = json.dumps(
            results,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

        prompt = f"""
Pergunta do usuário:
{question}

SQL executado:
{sql}

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
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("A pergunta original não pode estar vazia.")

        if not isinstance(queries, list) or not queries:
            raise ValueError("É necessário informar pelo menos uma consulta.")

        payload = []

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

            payload.append(
                {
                    "id": query.get("id", index),
                    "objective": str(query.get("objective", "")),
                    "question": str(query.get("question", "")),
                    "status": status,
                    "total_results": query.get(
                        "total_results",
                        len(results),
                    ),
                    "results": self._limit_rows(results),
                    "error": (str(query["error"]) if query.get("error") else None),
                }
            )

        successful_results = [
            item for item in payload if item["status"] == "success" and item["results"]
        ]

        # Não consumir uma chamada de IA quando não existem
        # resultados úteis para interpretar.
        if not successful_results:
            return self._build_no_results_answer(payload)

        serialized_payload = self._serialize_payload(payload)

        prompt = f"""
PERGUNTA ORIGINAL DO USUÁRIO:
{question}

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
9. Cada consulta informa `total_results`, `provided_results`,
   `omitted_results` e `results_truncated`.
10. Quando `results_truncated` for true, considere os registros
    enviados como uma amostra parcial dos resultados.
11. Nunca afirme que meses, categorias ou status ausentes da
    amostra nao existem na base; eles podem estar entre os
    registros omitidos da sintese.
12. Nao apresente totais ou tendencias do conjunto completo
    quando os resultados estiverem truncados. Restrinja as
    conclusoes aos registros enviados e identifique a limitacao.
13. Se os resultados forem insuficientes, informe isso.
14. Nao apresente SQL.
15. Evite repeticoes e introducoes genericas.

Produza uma resposta integrada, organizada e objetiva,
destacando os principais achados de cada objetivo.
"""

        return self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self._system_prompt(multi=True),
            max_output_tokens=900,
        )

    @staticmethod
    def _limit_rows(results: list[dict]) -> list[dict]:
        """Limita linhas, colunas e tamanho de valores para síntese."""

        limited = []

        for row in results[:MAX_SYNTHESIS_ROWS]:
            limited_row = {}

            for column, value in list(row.items())[:MAX_SYNTHESIS_COLUMNS]:
                if isinstance(value, str):
                    value = value[:MAX_CELL_LENGTH]

                limited_row[str(column)] = value

            limited.append(limited_row)

        return limited

    @staticmethod
    def _serialize_payload(payload: list[dict]) -> str:
        """
        Limita o contexto enviado à IA sem modificar
        os resultados completos armazenados pelo agente.
        """

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

        while len(serialized) > MAX_SYNTHESIS_PAYLOAD_CHARS:
            candidates = [item for item in payload if item["results"]]

            if not candidates:
                break

            largest = max(
                candidates,
                key=lambda item: len(item["results"]),
            )

            largest["results"].pop()

            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )

        for item in payload:
            provided_results = len(item["results"])
            omitted_results = max(
                item["total_results"] - provided_results,
                0,
            )
            item["provided_results"] = provided_results
            item["omitted_results"] = omitted_results
            item["results_truncated"] = omitted_results > 0

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )

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

    @staticmethod
    def _system_prompt(multi: bool = False) -> str:
        """Retorna as instruções do analista."""

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
