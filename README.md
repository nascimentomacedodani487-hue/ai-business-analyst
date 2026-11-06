# AI Business Analyst

Agente de IA que responde perguntas de negócio **em linguagem natural** sobre dados de um marketplace de delivery: gera o SQL, executa com segurança, mostra a tabela/gráfico e explica o resultado.

> **Status: em construção.** Esta seção é atualizada a cada etapa concluída; nada abaixo é listado como pronto sem estar implementado e testado.

## Roteiro

| Etapa | Descrição | Status |
|---|---|---|
| 1 | Estrutura do projeto, dados sintéticos reproduzíveis, banco DuckDB somente leitura, testes | ✅ concluída |
| 2 | Guardrails de SQL (só `SELECT`, bloqueio de funções perigosas) + testes | ⏳ planejada |
| 3 | Integração com LLMs (Gemini, Claude, GPT) por adaptadores e prompts versionados | ⏳ planejada |
| 4 | Conjunto de avaliação (perguntas + SQL esperado) e medição de acerto por modelo | ⏳ planejada |
| 5 | Interface Streamlit (pergunta → SQL → tabela → gráfico → explicação) | ⏳ planejada |
| 6 | Deploy online e demonstração | ⏳ planejada |

## Dados

Os dados são **100% sintéticos** (`src/ai_analyst/data/generate.py`), gerados com seed fixa: 200 restaurantes, 5.000 clientes, 300 entregadores e 60.000 pedidos de 2025 em 5 cidades.
Foram injetados padrões de propósito (pico no almoço/jantar, mais pedidos de sexta a domingo, chuva aumentando tempo de entrega e cancelamentos), e os testes verificam que esses padrões existem. **Qualquer "insight" encontrado reflete essas regras, não o mundo real.**

## Segurança

O SQL gerado por LLM roda numa conexão DuckDB **somente leitura**, **sem acesso a arquivos externos** (`read_csv`, `COPY`, `ATTACH`) e com a configuração **travada**. Testes automatizados comprovam esses bloqueios.

## Como rodar

Pré-requisito: Python 3.11+.

```powershell
git clone <url-do-repositorio>
cd ai-business-analyst
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

python -m ai_analyst.data.db      # gera data/delivery.duckdb
pytest                            # roda os testes
```

## Estrutura

```
src/ai_analyst/data/   generate.py (dados sintéticos) · db.py (banco + conexão segura) · schema.py (documentação do schema)
tests/                 testes do gerador e da segurança da conexão
data/                  banco gerado localmente (ignorado pelo Git)
```
