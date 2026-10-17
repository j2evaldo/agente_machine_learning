### Multi-Agent Statistical, Econometric & Feature Engineering Framework (OpenCode)
Framework autônomo, modular e de nível enterprise para Ciência de Dados, Engenharia de Features e Econometria baseado em agentes no terminal (OpenCode). Projetado para unir rigor estatístico, inferência causal, governança de dados e impacto financeiro direto nas decisões de negócio.

--------------------------------------------------------------------------------

#### **1. Proposta do Projeto & Ciclo de Vida Analítico**
O framework estrutura o ciclo de vida analítico (CRISP-DM) com governança estrita e isolamento de contexto entre subagentes:
1. **Discovery & Scoping:** O `@orchestrator` sabatina o usuário sobre os 5 pilares de negócio, restrições operacionais e Unit Economics (matriz de custos FP/FN) antes de liberar qualquer execução de código, culminando no **Termo de Enquadramento Analítico (Scoping Charter)**.
2. **Auditoria Estrutural & Triagem (EDA):** O `@eda_specialist` realiza auditoria de integridade, prevenção rigorosa de *target leakage*, testes de hipóteses formais (Mann-Whitney U, $\chi^2$) e triagem de preditores via Information Value (IV) e Weight of Evidence (WoE).
3. **Engenharia de Atributos & Isolamento (Fit/Transform):** O `@feature_engineering_specialist` constrói atributos comportamentais (janelas móveis de velocidade, OOF Target Encoding, binning monotônico de WoE e transformações Yeo-Johnson/Box-Cox) garantindo o aprendizado exclusivo no conjunto de Treino (`.fit()`) e aplicação em Teste/OOT (`.transform()`).
4. **Modelagem Preditiva Avançada & Calibração:** O `@classification_specialist` (CatBoost, Boruta/ARFS, TreeSHAP) e o `@logit_specialist` (GLM, Odds Ratios, AME) executam o **Benchmark Progressivo em 3 Estágios** e calibração de probabilidade.
5. **Econometria Clássica & Causalidade:** O `@linear_regression_wooldridge` (OLS, premissas de Gauss-Markov, erros HC3) e o `@causal_specialist` (Double Machine Learning, Propensity Score, DAGs) estimam elasticidades, isolam o efeito causal ($P(Y|do(T))$) e otimizam preço ($P^*$).

--------------------------------------------------------------------------------

#### **2. Arquitetura do Sistema de Agentes**
##### **Subagentes (.opencode/agents/)**
* **`@orchestrator` (Analytics Tech Lead & Discovery)**
  * Condução dos 5 Pilares de Discovery de Negócio e emanação do *Scoping Charter*.
  * Garantia do split em 3 vias (`D_train`, `D_test`, `D_val_oot`) e direcionamento do DAG multiagentes.
* **`@eda_specialist` (Auditoria & Quality Control)**
  * Diagnóstico read-only de integridade, prevenção de *leakage* (IV > 0.50, $|r| > 0.85$) e testes de hipóteses.
* **`@feature_engineering_specialist` (Transformação & Pipelines)**
  * Engine de atributos comportamentais (velocidade 30/180d, EWMA), OOF Target Encoding e binning de WoE.
  * Preservação do isolamento temporal via abstração `fit/transform` e suporte a $n-1$ dummies para modelos paramétricos.
* **`@classification_specialist` (GBDT & Inteligência Comercial)**
  * Modelagem CatBoost com seleção Boruta/ARFS, tuning Optuna e explicabilidade SHAP (Beeswarm e Waterfall).
  * Otimização de threshold financeiro ($c^*$), ordenação $Top-K$ e calibração de probabilidade (Isotônica/Platt).
* **`@logit_specialist` (GLM & Explicabilidade Econométrica)**
  * Modelagem paramétrica com preservação das probabilidades reais (sem resampling artificial), Odds Ratios, AME e controle de VIF < 10.
* **`@causal_specialist` (Inferência Causal & Counterfactuals)**
  * Isolamento de efeito de tratamento via Double Machine Learning (DML), Propensity Score Matching e testes de refutação (placebo).
* **`@linear_regression_wooldridge` (Econometria OLS)**
  * Validação das 6 premissas de Gauss-Markov (BLUE), erros robustos HC3 e cálculo de elasticidades de preço/demanda.

--------------------------------------------------------------------------------

#### **3. Protocolos Globais de Governança e Runtime**

##### **3.1 Infraestrutura (Python 3.12 + `uv`)**
* **Runtime:** Obrigatoriamente **Python 3.12.***.
* **Gerenciador de Dependências:** `uv` (Astral). Proibido uso direto de `pip` ou `conda`.
* **PEP 723 Inline Script Metadata:** Todo script Python autônomo deve conter o cabeçalho de dependências para execução reprodutível instantânea via `uv run`.

##### **3.2 Partição de Dados em 3 Vias (3-Way Split)**
```text
  Dados Brutos Totais (D)
   │
   ├──────► Treino (D_train) ─────► [fit() das Transformações & Fit dos Parâmetros]
   │
   ├──────► Teste (D_test)   ─────► [transform() & Optuna / Calibração / Threshold Opt (c*)]
   │
   └──────► Validação OOT (D_val) ─► [transform() & Avaliação Final de ROI e Estabilidade]
```

##### **3.3 Benchmark Progressivo em 3 Estágios**
Toda modelagem deve reportar a matriz de desempenho no conjunto Out-Of-Time (`D_val_oot`):
1. **Estágio 1 (Vanilla Model):** Baseline simples (ex: Regressão Logística/Árvore rasa) com tratamento mínimo.
2. **Estágio 2 (Modelo Robusto):** Algoritmo avançado (ex: CatBoost) com tuning de hiperparâmetros (isolar $\Delta_{\text{Capacity}}$).
3. **Estágio 3 (FE Model):** Modelo robusto alimentado pela matriz enriquecida pelo `@feature_engineering_specialist` (isolar $\Delta_{\text{FE}}$).

--------------------------------------------------------------------------------

#### **4. Fluxo de Execução Multiagentes (DAG)**
```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                GRAFO DE EXECUÇÃO MULTIAGENTES                                   │
├─────────┬───────────────────────────────┬───────────────────────────────────────────────────────┤
│ Fase    │ Agente                        │ Responsabilidade Principal                            │
├─────────┼───────────────────────────────┼───────────────────────────────────────────────────────┤
│ Fase 1  │ @orchestrator                 │ Discovery, Scoping Charter e Garantia do Split 3-Ways │
│ Fase 2  │ @eda_specialist               │ Auditoria Read-Only, Detecção de Leakage e IV         │
│ Fase 3  │ @feature_engineering_spec.    │ Pipelines Fit/Transform, Rolling Windows e OOF TE     │
│ Fase 4  │ Subagente de Modelagem        │ Benchmark 3 Estágios (Vanilla, Robusto, FE) & Calib   │
│ Fase 5  │ @orchestrator                 │ Síntese Executiva, Explicabilidade e ROI Financeiro   │
└─────────┴───────────────────────────────┴───────────────────────────────────────────────────────┘
```

--------------------------------------------------------------------------------

#### **5. Estrutura do Repositório**
```text
.
├── .opencode/
│   ├── instructions.md                 # Governança global de código, estilo e runtime
│   └── agents/
│       ├── orchestrator.md             # Discovery, Scoping Charter e Governança do DAG
│       ├── eda_specialist.md           # Auditoria de dados, Leakage e Testes Bivariados
│       ├── feature_engineering_spec.md  # Engine de Atributos, Fit/Transform e Rolling Windows
│       ├── classification_specialist.md# CatBoost, Boruta/ARFS, TreeSHAP e Calibração
│       ├── logit_specialist.md         # GLM, Odds Ratios, AME e Controle de VIF
│       ├── causal_specialist.md        # Double ML, Propensity Score e Testes Cautelares
│       └── linear_regression_wooldridge.md # OLS Econométrico, Gauss-Markov e Elasticidades
├── src/
│   ├── __init__.py
│   ├── skills/
│   └── models/
├── notebooks/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── pyproject.toml
└── README.md
```

--------------------------------------------------------------------------------

#### **6. Instruções de Execução**
##### **Pré-requisitos & Instalação com `uv`**
```bash
# Criação do ambiente virtual em Python 3.12
uv venv --python 3.12

# Sincronização e instalação de dependências
uv sync
```

##### **Inicialização e Exemplo de Uso**
```bash
opencode
```
```text
> @orchestrator Preciso estruturar um modelo de churn para a base de assinantes com foco em acionamento comercial no CRM.
```

--------------------------------------------------------------------------------

#### **7. Testes e Qualidade de Código**
```bash
# Suíte de testes com uv
uv run pytest -v

# Checagem e formatação de código
uv run ruff check .
```
