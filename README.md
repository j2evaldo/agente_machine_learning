# Multi-Agent Statistical & Econometric Framework (OpenCode)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenCode](https://img.shields.io/badge/OpenCode-Harness%20AI-blueviolet?style=flat)](https://opencode.ai)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Econometrics%20%26%20GLM-4B8BBE?style=flat)](https://www.statsmodels.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![uv](https://img.shields.io/badge/Packaging-uv-261230?style=flat&logo=astral)](https://github.com/astral-sh/uv)

Framework autônomo e modular de Ciência de Dados e Econometria baseado em agentes no terminal (OpenCode). Projetado para unir rigor estatístico, inferência causal, governança em modelagem e impacto financeiro nas decisões de negócio.

---

## 1. Proposta do Projeto

O framework estrutura o ciclo de vida analítico (CRISP-DM) com governança estrita e isolamento de contexto entre subagentes:

1. **Discovery & Scoping:** O `@orchestrator` sabatina o usuário sobre dores de negócio, janelas do alvo, restrições operacionais e Unit Economics (matriz de custos FP/FN) antes de liberar qualquer execução.
2. **Auditoria Estrutural & Triagem (EDA):** Verificação de integridade, prevenção de *target leakage*, testes de hipóteses formais (Mann-Whitney, $\chi^2$) e triagem de preditores via Information Value (IV) e Weight of Evidence (WoE).
3. **Modelagem Paramétrica & GLM:** Regressão Logística com checagem de linearidade no logit por decis, codificação $n-1$, Efeitos Marginais Médios (AME em p.p.) e otimização financeira de threshold ($c^*$).
4. **Econometria Clássica & Causalidade:** Regressão linear OLS (Wooldridge), validação das hipóteses de Gauss-Markov, matriz de covariância robusta (Huber-White HC3), cálculo de elasticidades e otimização de precificação ($P^*$).

---

## 2. Arquitetura do Sistema de Agentes & Skills

### Subagentes (`.opencode/agents/`)
- **`@orchestrator` (Analytics Tech Lead & Discovery)**
  - Condução dos 5 Pilares de Discovery de Negócio.
  - Formalização do *Scoping Charter* (Portão de Aprovação).
  - Roteamento e síntese de relatórios executivos com impacto financeiro em R$.

- **`@logit_specialist` (GLM & Classificação Comercial)**
  - Auditoria de premissas (logit empírico, VIF, categorias de referência).
  - Tradução de Odds Ratios e Efeitos Marginais (AME em pontos percentuais).
  - Avaliação nos 3 Objetivos: **Decidir** (Threshold $c^*$), **Ordenar** (Lift/KS) e **Estimar** (Brier Score/Calibração).

- **`@linear_regression_wooldridge` (Econometria & Causalidade)**
  - Validação de Gauss-Markov (BLUE), testes de Breusch-Pagan, White e Ramsey RESET.
  - Correção obrigatória de heterocedasticidade com erros robustos HC3.
  - Matriz de elasticidades comerciais e otimizador de preço ótimo ($P^*$).

### Skills Compartilhadas (`src/skills/`)
- **`eda_tools.py`**: Funções modulares de auditoria de nulos, detecção de *leakage*, cálculo de WoE/IV, testes bivariados e geração de dashboards 2x2.

---

## 3. Diretrizes de Decisão por Paradigma Comercial

```text
                               ┌──────────────────────────────────────────────┐
                               │         DEMANDA / DOR DE NEGÓCIO             │
                               └──────────────────────┬───────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┴────────────────────────────────────────────┐
         ▼                                                                                         ▼
┌─────────────────────────────────┐                                                       ┌─────────────────────────────────┐
│     ALAVANCAS CAUSAIS & PREÇO   │                                                       │     PROBABILIDADES & EVENTOS    │
│ (Elasticidade, Ceteris Paribus) │                                                       │ (Conversão, Churn, Fraude, Lead)│
└────────────────┬────────────────┘                                                       └────────────────┬────────────────┘
                 │                                                                                         │
                 ▼                                                                                         ▼
┌─────────────────────────────────┐                                                       ┌─────────────────────────────────┐
│ @linear_regression_wooldridge   │                                                       │       @logit_specialist         │
│ • Elasticidade Preço/Demanda    │                                                       │ • DECIDIR (Threshold c*, Unit   │
│ • Retornos Decrescentes Marketing│                                                      │   Economics: FP/FN)             │
│ • Gauss-Markov & Erros HC3      │                                                       │ • ORDENAR (Lift decil, KS, AUC) │
└─────────────────────────────────┘                                                       │ • ESTIMAR (Brier Score, ECE)    │
                                                                                          └─────────────────────────────────┘
```

---

## 4. Estrutura de Arquivos

```text
.
├── .opencode/
│   └── agents/
│       ├── orchestrator.md
│       ├── logit_specialist.md
│       └── linear_regression_wooldridge.md
├── src/
│   ├── __init__.py
│   ├── skills/
│   │   ├── __init__.py
│   │   └── eda_tools.py
│   └── models/
│       ├── __init__.py
│       ├── logit_model.py
│       └── linear_model.py
├── tests/
│   ├── __init__.py
│   └── test_eda_tools.py
├── notebooks/
├── data/
│   └── raw/
├── AGENTS.md
├── pyproject.toml
└── README.md
```

---

## 5. Instruções de Execução

### Pré-requisitos & Instalação

Gerenciamento de dependências via `uv`:

```bash
# Sincronizar o ambiente virtual e dependências
uv sync
```

### Inicialização do OpenCode

Inicie o assistente a partir da raiz do repositório:

```bash
opencode
```

### Exemplo de Uso

Inicie sempre pelo orquestrador para realizar o enquadramento de negócio:

```text
> @orchestrator Preciso estimar a elasticidade-preço da nossa linha de produtos e recomendar o preço ótimo de venda.
```

O `@orchestrator` abrirá a sabatina de Discovery antes de delegar a análise econométrica para o `@linear_regression_wooldridge`.

---

## 6. Testes e Qualidade de Código

```bash
# Execução da suíte de testes
uv run pytest -v

# Validação e formatação de código
uv run ruff check .
```