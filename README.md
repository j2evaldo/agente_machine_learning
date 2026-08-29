# Multi-Agent Statistical Machine Learning Framework (OpenCode)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenCode](https://img.shields.io/badge/OpenCode-Harness%20AI-blueviolet?style=flat)](https://opencode.ai)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Econometrics%20%26%20GLM-4B8BBE?style=flat)](https://www.statsmodels.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![uv](https://img.shields.io/badge/Packaging-uv-261230?style=flat&logo=astral)](https://github.com/astral-sh/uv)

Framework autonomo e modular de Ciencia de Dados baseado em agentes no terminal (OpenCode), desenhado para unir rigor estatistico, econometria, governanca de modelagem e alinhamento a objetivos de negocio.

---

## 1. Proposta do Projeto

O objetivo deste framework e mitigar falhas comuns no ciclo de modelagem, tais como violacoes de premissas estatisticas, escopo de negocio mal delimitado e falta de interpretabilidade em setores regulados.

A estrutura implementa agentes especializados com contexto efemero e isolamento de tarefas, estruturados no fluxo CRISP-DM:

1. **Entendimento e Escopo de Negocio:** Mapeamento de dores, custos de erro (FP/FN) e restricoes antes da escrita de codigo.
2. **Analise Exploratoria e Inferencial:** Testes de hipoteses bivariados (Qui-Quadrado, Mann-Whitney U) e calculo de Information Value (IV).
3. **Auditoria de Premissas Estatisticas:** Validacao de linearidade no logit por decis, controle de multicolinearidade (VIF) e codificacao estrita de variaveis dummy (n-1).
4. **Otimizacao por Objetivo de Negocio:** Diagnostico voltado para Decisao (threshold otimo), Ordenacao (ROC-AUC, KS, Lift) ou Estimacao (RMSE, Calibracao, Valor Esperado).

---

## 2. Arquitetura do Sistema de Agentes

Fluxo de trabalho:

- **@orchestrator (Lead Tech)**
  - Business Scoping e Dores
  - Formulacao Matematica
  - Roteamento e Planejamento
  - *Delega para:* @eda_specialist, @logit_specialist e @ml_scientist

- **@eda_specialist (Inferencia e Qualidade)**
  - Testes t e Mann-Whitney U
  - Teste Qui-Quadrado de Independencia
  - Information Value (IV) e Weight of Evidence (WoE)
  - Auditoria de nulos, cardinalidade e distribuicoes

- **@logit_specialist (Modelagem Parametrica e GLM)**
  - Linearidade com logit por decis
  - Codificacao de dummies (regra n-1)
  - Odds Ratios com intervalos de confianca
  - Otimizacao de threshold financeiro

- **@ml_scientist (Modelos Nao-Lineares)**
  - Modelos de Gradient Boosting (CatBoost, LightGBM)
  - Validacao cruzada e ajuste de hiperparametros
  - Analises comparativas de benchmark

---

## 3. Diretrizes de Otimizacao por Objetivo de Negocio

- **Decidir:** Foco em acao individual binaria (aprovar/recusar, intervir/nao intervir). Metrica principal: Threshold financeiro otimo maximizando lucro esperado total.
- **Ordenar:** Foco em capacidade operacional limitada (acionamento em ranking/fila). Metrica principal: Curva ROC-AUC, Estatistica KS e Cumulative Lift nos decis superiores.
- **Estimar:** Foco em avaliacao atuarial e perda financeira esperada da carteira. Metrica principal: Calibracao de probabilidade, Brier Score e RMSE.

---

## 4. Estrutura de Arquivos

```text
.
├── .opencode/
│   └── agents/
│       ├── orchestrator.md
│       ├── eda_specialist.md
│       ├── logit_specialist.md
│       └── ml_scientist.md
├── AGENTS.md
├── pyproject.toml
├── src/
├── notebooks/
└── tests/
```

---

## 5. Instrucoes de Execucao

### Pre-requisitos

Instalacao do gerenciador de dependencias uv:

```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
uv sync
```

### Inicializacao do Ambiente

Abra a sessao do OpenCode na raiz do projeto:

```bash
opencode
```

### Exemplo de Uso

Inicie o contato acionando o orquestrador para definir o escopo:

```text
> @orchestrator Preciso construir um modelo de concessao de credito para clientes PJ.
```

O orquestrador fara as perguntas de escopo estrategico antes de coordenar as analises estatisticas e a modelagem.

---

## 6. Testes e Validacao de Codigo

```bash
uv run pytest -v
uv run ruff check .
```