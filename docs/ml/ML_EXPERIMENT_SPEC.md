# AI Business Analyst — Especificação Técnica de Machine Learning

**Versão:** 0.1.0
**Status:** Proposta técnica — pré-implementação
**Dataset:** Synthetic Delivery Dataset · 2025
**Banco:** `data/delivery.duckdb`

---

# 1. Objetivo

Definir os fundamentos técnicos para incorporar Machine Learning ao AI Business Analyst.

O módulo deverá oferecer experimentação reproduzível, treinamento, avaliação, comparação, versionamento e disponibilização de modelos.

A primeira implementação deverá utilizar os dados existentes, preservando o banco original e evitando vazamento de informações.

Os resultados atuais são experimentos realizados sobre dados sintéticos. Não representam evidências de desempenho em operações reais.

# 2. Estado atual dos dados

## 2.1. Estrutura confirmada

| Tabela | Registros |
|---|---:|
| orders | 60.000 |
| customers | 5.000 |
| restaurants | 200 |
| couriers | 300 |

Características:

- 57.828 pedidos entregues.
- 2.172 pedidos cancelados.
- Taxa de cancelamento: 3,62%.
- Período: 01/01/2025 a 31/12/2025.
- Cinco cidades.
- 365 dias observados por cidade.
- Seed de geração: 42.

A equivalência entre o DuckDB local e o gerador foi confirmada para as quatro tabelas. A tabela `customers` inicialmente apresentou divergência por representação dos tipos de data, mas os 5.000 registros foram posteriormente comparados e confirmados como idênticos após normalização.

## 2.2. Colunas relevantes

**orders:**

- order_id
- customer_id
- restaurant_id
- courier_id
- order_ts
- city
- status
- items_total
- delivery_fee
- discount
- payment_method
- distance_km
- prep_minutes
- delivery_minutes
- is_raining
- customer_rating

**restaurants:**

- restaurant_id
- name
- category
- city
- avg_rating

**couriers:**

- courier_id
- city
- vehicle

**customers:**

- customer_id
- city
- signup_date

# 3. Características do gerador sintético

Os dados foram construídos com padrões deliberados:

- Sexta, sábado e domingo possuem peso maior na geração de pedidos.
- O horário de almoço e jantar apresenta maior concentração.
- Chuva aumenta a duração de entrega e a probabilidade de cancelamento.
- Categorias de restaurantes influenciam tempo de preparo e valor dos pedidos.
- Tipos de veículo possuem velocidades definidas.
- Avaliações tendem a diminuir quando o tempo total aumenta.

A duração de entrega é gerada principalmente por distância, velocidade atribuída ao veículo, chuva, horário de pico e ruído aleatório.

A probabilidade de cancelamento é gerada por uma regra que considera chuva e tempo de preparo.

Essas relações são importantes para construir experimentos controlados, mas não devem ser confundidas com relações comprovadas em operações reais.

# 4. Primeiro experimento: previsão de demanda

## 4.1. Definição do problema

Prever a quantidade de pedidos por cidade e dia.

**Variável-alvo:**

Quantidade de pedidos criados em cada cidade durante cada dia, incluindo entregues e cancelados.

A estrutura resultante possui:

- 5 cidades.
- 365 dias.
- 1.825 observações.
- Nenhum dia sem pedidos no dataset atual.

A previsão deverá utilizar apenas informações disponíveis antes do período previsto.

## 4.2. Baselines avaliados

Protocolo inicial:

- Treinamento: 01/01/2025 a 05/11/2025.
- Teste: 06/11/2025 a 31/12/2025.
- 309 dias de treinamento.
- 56 dias de teste.
- 1.545 observações de treinamento.
- 280 observações de teste.

Resultados:

| Baseline | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Média por cidade e dia da semana | 4,410 | 5,709 | 13,457% |
| Sazonal ingênuo (semana anterior) | 6,561 | 8,323 | 20,017% |

A média histórica por cidade e dia da semana apresentou redução de aproximadamente 32,8% no WAPE em relação ao baseline sazonal ingênuo.

O resultado ainda precisa ser confirmado com múltiplas janelas temporais.

## 4.3. Modelos candidatos

Primeira comparação proposta:

1. Média histórica por cidade e dia da semana.
2. Sazonal ingênuo.
3. Regressão de Poisson (GLM).
4. Regressão binomial negativa, caso a dispersão dos dados justifique.
5. Modelo com features de calendário e interações entre cidade e dia da semana.

Não adicionar boosting antes de avaliar os baselines.

## 4.4. Features candidatas

- Cidade.
- Dia da semana.
- Dia do ano, apenas para experimentos controlados.
- Tendência temporal.
- Histórico de pedidos disponível antes da previsão.
- Variáveis temporais derivadas do calendário.

Não utilizar informações futuras ou agregações calculadas com dados posteriores ao instante de previsão.

Possível extensão futura: previsão por cidade e hora, caso o protocolo e a granularidade sejam avaliados separadamente.

## 4.5. Métricas

- MAE.
- RMSE.
- WAPE.

A comparação deve ser feita globalmente e por cidade.

Não utilizar divisão aleatória para avaliar previsão temporal.

# 5. Segundo experimento: regressão de duração da entrega

## 5.1. Definição do problema

Prever `delivery_minutes` para pedidos entregues.

Existem 57.828 registros com duração conhecida.

O significado do alvo é o tempo de entrega após a saída do restaurante.

**Limitação:** o dataset não possui timestamp de atribuição do entregador, preparação concluída ou saída do restaurante.

Portanto, a previsão ainda não representa um ETA operacional completo.

O modelo será experimental até estabelecermos um instante de inferência coerente.

## 5.2. Experimentos já realizados

Separação temporal inicial:

- Treinamento: janeiro a 05/11/2025.
- Teste: 06/11/2025 a 31/12/2025.
- 49.012 registros de treinamento.
- 8.816 registros de teste.

Resultados:

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| Média constante | 6,0680 | 8,0448 | aproximadamente 0 |
| Linear aditiva com chuva | 2,7350 | 3,5604 | 0,8041 |
| Interações veículo × distância, sem chuva | 2,7861 | 3,5237 | 0,8081 |
| Interações veículo × distância, com chuva | 2,2870 | 2,8570 | 0,8739 |

O modelo com interações e chuva reduziu o MAE em aproximadamente 62,3% em relação à média constante.

## 5.3. Validação temporal expansiva

Foram realizadas quatro janelas:

| Teste | MAE sem chuva | MAE com chuva |
|---|---:|---:|
| Setembro | 2,7587 | 2,2742 |
| Outubro | 2,7831 | 2,2665 |
| Novembro | 2,8628 | 2,2811 |
| Dezembro | 2,7560 | 2,2978 |

Resultado consolidado:

| Métrica | Sem chuva | Com chuva |
|---|---:|---:|
| MAE | 2,7895 | 2,2798 |
| RMSE | 3,5227 | 2,8541 |
| P90 absoluto | 5,7552 | 4,6856 |

Foram avaliadas 19.450 previsões, em quatro janelas temporais.

A inclusão da chuva reduziu o MAE em todas as quatro janelas, com melhoria média de 18,27%.

## 5.4. Resultados segmentados

O modelo com chuva apresentou melhora em todas as cidades, categorias de veículo e condições climáticas avaliadas.

Exemplos de MAE:

| Segmento | Sem chuva | Com chuva |
|---|---:|---:|
| Belo Horizonte | 2,601 | 2,284 |
| Curitiba | 2,737 | 2,318 |
| Recife | 2,871 | 2,282 |
| Rio de Janeiro | 2,821 | 2,290 |
| São Paulo | 2,830 | 2,274 |
| Bicicleta | 3,222 | 2,368 |
| Carro | 2,883 | 2,372 |
| Moto | 2,676 | 2,256 |

## 5.5. Limitação importante da variável chuva

`is_raining` é constante por cidade e dia, conforme confirmado nas 1.825 combinações de cidade e data.

O campo representa o estado meteorológico realizado naquele dia, não uma previsão meteorológica registrada no instante de inferência.

Portanto:

- Modelo sem chuva: baseline operacional conservador.
- Modelo com chuva: experimento retrospectivo com informação meteorológica realizada.

O resultado de 2,2798 minutos não deve ser anunciado como desempenho operacional disponível sem restrições.

Uma evolução futura deverá integrar previsão meteorológica com timestamp de consulta.

## 5.6. Features candidatas

| Feature | Regra |
|---|---|
| distance_km | Condicional à disponibilidade da rota |
| vehicle | Somente após atribuição do entregador |
| order_ts | Disponível no pedido, mas não equivale à saída |
| is_raining | Retrospectiva no dataset atual |
| prep_minutes | Somente quando o preparo estiver concluído e disponível no instante definido |
| delivery_minutes | Variável-alvo |
| customer_rating | Informação posterior, proibida |
| status | Utilizada para selecionar pedidos entregues |
| IDs | Não utilizar diretamente |

O modelo inicial utilizará distância, categoria do veículo, horário e suas interações.

Chuva será mantida apenas como comparação retrospectiva até conseguirmos uma feature meteorológica operacional.

## 5.7. Métricas

- MAE.
- RMSE.
- R².
- P90 do erro absoluto.
- Erros por cidade.
- Erros por veículo.
- Erros por condição climática.

Também devemos verificar previsões negativas, distribuição dos resíduos e possíveis intervalos preditivos.

# 6. Terceiro experimento: classificação de cancelamento

## 6.1. Definição

Prever a probabilidade de cancelamento de um pedido.

Distribuição atual:

- 2.172 cancelados.
- 57.828 entregues.
- Prevalência positiva: 3,62%.

O problema possui forte desbalanceamento.

## 6.2. Problemas identificados

Todos os pedidos cancelados apresentam:

- `delivery_minutes` nulo.
- `customer_rating` nulo.

Além disso:

- 23.068 entregas concluídas também não possuem avaliação.
- `courier_id` está preenchido inclusive nos pedidos cancelados.
- O momento da atribuição do entregador não está registrado.
- `prep_minutes` influencia diretamente a geração do cancelamento.

O classificador não deverá utilizar informações posteriores ao instante da previsão.

## 6.3. Definição preliminar das features

Para previsão no momento de criação do pedido, considerar:

- order_ts.
- city.
- items_total.
- delivery_fee.
- discount.
- payment_method.
- distance_km, caso já esteja disponível.
- categoria do restaurante.
- avaliação média do restaurante.
- previsão meteorológica, se disponível naquele instante.

Não utilizar:

- prep_minutes realizado.
- delivery_minutes.
- customer_rating.
- courier_id ou vehicle sem comprovação de disponibilidade no instante previsto.

A classificação continuará experimental até demonstrarmos que existe sinal preditivo útil com as informações disponíveis no momento de decisão.

## 6.4. Modelos candidatos

- Logistic Regression.
- Random Forest somente se justificado.
- Possíveis modelos de boosting em experimentos futuros.

Métricas:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Brier Score.
- Calibração.
- Matriz de confusão.
- Avaliação por custo de erro.

Não utilizar somente acurácia.

# 7. Contrato de disponibilidade das features

O módulo de ML deverá respeitar a disponibilidade temporal das variáveis.

Existem três estágios conceituais:

### Estágio A — criação do pedido

Disponíveis ou potencialmente disponíveis:

- Data e hora do pedido.
- Cidade.
- Restaurante selecionado.
- Itens e valores.
- Método de pagamento.
- Distância calculada, caso a rota já tenha sido estabelecida.
- Condição meteorológica prevista, caso exista integração apropriada.

### Estágio B — entregador atribuído / saída do restaurante

Potencialmente disponíveis:

- Tipo de veículo.
- Distância.
- Informações de preparo concluído.
- Hora efetiva de saída.
- Condição meteorológica observada no instante.

O banco atual não registra todos esses eventos. As features deverão ser tratadas como condicionais.

### Estágio C — entrega concluída

Informações posteriores:

- Duração real.
- Status final.
- Avaliação do cliente.

Essas variáveis podem servir como alvo, histórico ou avaliação, mas não como preditoras de eventos anteriores.

**Regra:** toda feature deverá declarar o instante em que fica disponível e sua origem. O treinamento deverá recusar features incompatíveis com o instante de previsão configurado.

# 8. Nova versão do dataset

Não modificar o dataset v1 nem os números históricos.

Criar futuramente uma versão v2 específica para ML, preservando o dataset original.

Possíveis novas informações:

- courier_assigned_ts.
- restaurant_ready_ts.
- courier_departure_ts.
- delivery_completed_ts.
- previsões meteorológicas com timestamp.
- Mais anos de observações.
- Variação temporal e geográfica mais rica.
- Anomalias sintéticas documentadas.

Os dados gerados devem ser identificados explicitamente como sintéticos.

# 9. Arquitetura preliminar do módulo ML

Proposta de pacote:

`src/ai_analyst/ml/`

Organização conceitual:

- `features.py`: definição e validação de features.
- `datasets.py`: preparação dos conjuntos de dados.
- `splitting.py`: validação temporal.
- `baselines.py`: modelos de referência.
- `regression.py`: regressão.
- `forecasting.py`: previsão de demanda.
- `classification.py`: classificação.
- `metrics.py`: métricas específicas de ML.
- `experiments.py`: execução de experimentos.
- `history.py`: histórico persistente.
- `artifacts.py`: versionamento e carregamento dos modelos.
- `schemas.py`: contratos de entrada e saída.

Treinamento deverá ser separado da inferência.

O treinamento será realizado por uma interface CLI.

A API deverá carregar somente modelos registrados e validados.

Persistência inicial sugerida:

- Manifesto JSON com configuração e métricas.
- Artefatos de modelos versionados.
- Histórico SQLite.
- Fingerprint do dataset.
- Seed.
- Features.
- Hiperparâmetros.
- Janela temporal.
- Versões de dependências.
- Métricas e resultados.

Scikit-learn poderá ser introduzido como dependência opcional, em um grupo `[ml]` do `pyproject.toml`.

Não instalar bibliotecas até concluir o contrato e a arquitetura.

# 10. Avaliação e reprodutibilidade

Cada experimento deverá registrar:

- Identificador único.
- Dataset e fingerprint.
- Versão do gerador.
- Seed.
- Conjunto de features.
- Definição do alvo.
- Instante de previsão.
- Divisão temporal.
- Algoritmo.
- Hiperparâmetros.
- Métricas.
- Versão do código.
- Artefato gerado.

Nenhum modelo deverá ser declarado melhor apenas com base em uma divisão de teste.

A comparação deve utilizar as mesmas janelas e o mesmo conjunto de observações sempre que possível.

# 11. Roadmap técnico após a auditoria

1. Finalizar auditoria e contrato de features.
2. Especificar arquitetura do módulo ML.
3. Implementar baselines de regressão e previsão de demanda.
4. Implementar treinamento e experimentação reproduzível.
5. Avaliar classificação de cancelamento.
6. Implementar registro e histórico de modelos.
7. Adicionar explicabilidade.
8. Avaliar detecção de anomalias.
9. Desenvolver interface `/ml`.
10. Integrar e validar a API.
11. Adicionar testes automatizados.
12. Preparar segurança e deploy.

Clustering, semissupervisionado, aprendizado por reforço e modelos fundacionais tabulares ficam como possibilidades futuras, condicionadas a casos de uso e justificativa técnica.

# 12. Critérios para iniciar a implementação

Antes de desenvolver o primeiro módulo, precisamos decidir:

1. Qual será o instante exato de previsão da duração.
2. Quais features estão disponíveis nesse instante.
3. Se a primeira entrega será um experimento de regressão ou previsão de demanda.
4. Qual será o contrato de entrada e saída do modelo.
5. Como registrar e reproduzir experimentos.
6. Como diferenciar resultados operacionais de resultados retrospectivos ou simulados.

**Decisão recomendada:** começar pelo desenho do módulo de ML e pela previsão de duração como experimento controlado, preservando a previsão de demanda como segundo projeto.

O desenvolvimento só deverá começar depois da definição explícita dessas condições.

---

**Estado atual:** Etapa 1 — Auditoria dos dados, praticamente concluída.
**Próxima fase:** especificação da arquitetura e contrato operacional do módulo ML.

Nenhuma dependência de ML foi instalada, nenhum modelo foi treinado como funcionalidade do produto e nenhum arquivo de produção ou banco foi alterado durante esta auditoria.
