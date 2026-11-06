"""
Geração e validação de consultas SQL utilizando IA.
"""

from ai_analyst.agent.context import build_system_prompt
from ai_analyst.llm.base import LLMProvider
from ai_analyst.security.guardrails import validate_query


class SQLGenerator:
    """Transforma perguntas de negócio em SQL seguro."""

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

        # Alias temporário para compatibilidade com testes e integrações
        # que ainda acessam o atributo gemini.
        self.gemini = self.provider

        self.system_prompt = build_system_prompt()

    def generate(self, question: str) -> str:
        """Gera e valida uma consulta SQL."""

        if not question or not question.strip():
            raise ValueError(
                "A pergunta não pode estar vazia."
            )

        prompt = f"""
Transforme a pergunta abaixo em uma consulta SQL DuckDB.

PERGUNTA:
{question}

REGRAS:
- Retorne somente o código SQL.
- Não utilize blocos Markdown.
- Gere somente uma consulta SELECT.
- Utilize exclusivamente tabelas e colunas documentadas.
- Não invente informações.
- Utilize aliases claros para as colunas calculadas.
- Não gere explicações ou comentários.
"""

        response = self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self.system_prompt,
            max_output_tokens=512,
        )

        sql = response.strip()

        # Validar a consulta antes de devolvê-la.
        validated_sql = validate_query(sql)

        return validated_sql

    def correct_sql(
        self,
        question: str,
        failed_sql: str,
        error_message: str,
    ) -> str:
        """
        Solicita ao provedor a correção de uma consulta SQL
        que apresentou erro de sintaxe ou de execução.

        A consulta corrigida passa novamente pelos Guardrails.
        """

        if not question or not question.strip():
            raise ValueError(
                "A pergunta original não pode estar vazia."
            )

        if not failed_sql or not failed_sql.strip():
            raise ValueError(
                "A consulta SQL com falha não pode estar vazia."
            )

        if not error_message or not error_message.strip():
            raise ValueError(
                "A mensagem de erro não pode estar vazia."
            )

        prompt = f"""
A consulta SQL gerada anteriormente apresentou um erro de sintaxe ou de execução
no DuckDB. Analise o problema e produza uma consulta corrigida.

PERGUNTA ORIGINAL:
{question}

SQL QUE FALHOU:
{failed_sql}

MENSAGEM DO ERRO:
{error_message}

INSTRUÇÕES:
- Corrija o problema identificado no SQL.
- Preserve a intenção da pergunta original.
- Utilize exclusivamente tabelas e colunas documentadas
  no contexto do banco de dados.
- Verifique cuidadosamente aliases, nomes de tabelas,
  nomes de colunas, JOINs e agregações.
- Utilize sintaxe compatível com DuckDB.
- Não invente tabelas, colunas ou informações.
- Retorne somente uma consulta SQL SELECT.
- Não utilize blocos Markdown.
- Não gere explicações, comentários ou texto adicional.
"""

        response = self.provider.generate_response(
            user_prompt=prompt,
            system_prompt=self.system_prompt,
            max_output_tokens=512,
        )

        corrected_sql = response.strip()

        # Toda consulta corrigida precisa passar pelos Guardrails.
        return validate_query(corrected_sql)
