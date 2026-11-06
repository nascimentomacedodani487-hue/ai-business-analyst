
"""
Geração de respostas analíticas em linguagem natural.
"""

import json
import logging

from ai_analyst.llm.base import LLMProvider


logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Transforma resultados SQL em explicações de negócio."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        *,
        gemini: LLMProvider | None = None,
    ):
        """
        Inicializa o gerador com um provedor de IA.

        O argumento gemini permanece disponível por compatibilidade
        com versões anteriores. Novas implementações devem utilizar
        o argumento genérico provider.
        """

        if provider is not None and gemini is not None:
            if provider is not gemini:
                raise ValueError(
                    "Informe apenas um provedor de IA."
                )

        self.provider = provider or gemini

        if self.provider is None:
            raise ValueError(
                "Um provedor de IA deve ser informado."
            )

        # Alias temporário para compatibilidade com integrações antigas.
        self.gemini = self.provider

    def generate(
        self,
        question: str,
        sql: str,
        results: list[dict],
    ) -> str:
        """Interpreta resultados SQL e gera uma resposta."""

        # Não é necessário chamar a IA quando não existem resultados.
        if not results:
            return (
                "Não foram encontrados dados para responder "
                "à pergunta."
            )

        # JSON compacto reduz o tamanho do contexto enviado à API.
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
            system_prompt=(
                "Você é um analista de negócios especializado "
                "em interpretação de dados. "
                "Sua função é transformar resultados SQL em "
                "respostas claras, precisas e objetivas. "
                "Nunca invente informações."
            ),
            max_output_tokens=512,
        )
