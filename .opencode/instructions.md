# OPENCODE GLOBAL SYSTEM INSTRUCTIONS & RUNTIME GOVERNANCE

#### **1. INFRASTRUCTURE, ENVIRONMENT & RUNTIME STANDARDS**
* **Python Runtime:** Obrigatoriamente **Python 3.12.***. É estritamente proibido o uso de versões anteriores (Python 3.10/3.11).
* **Package & Environment Manager:** Obrigatoriamente **`uv`** (Astral). Proibido o uso direto de `pip`, `conda` ou Python global do sistema.
* **Comandos Padrão de Terminal (CLI Execution):**
  * Criação do Ambiente Virtual: `uv venv --python 3.12`
  * Instalação de Dependências: `uv pip install <pacote>` ou `uv add <pacote>`
  * Execução Isolada de Scripts: `uv run python <script.py>`
* **PEP 723 Inline Script Metadata:** Todo script Python autônomo gerado pelos agentes no OpenCode DEVE incluir o cabeçalho estandardizado de metadados inline:
  ```python
  # /// script
  # requires-python = ">=3.12"
  # dependencies = [
  #     "pandas>=2.2.0",
  #     "numpy>=1.26.0",
  #     "scikit-learn>=1.4.0",
  #     "catboost>=1.2.0",
  #     "optuna>=3.5.0",
  #     "shap>=0.44.0",
  #     "seaborn>=0.13.0",
  #     "statsmodels>=0.14.0",
  # ]
  # ///
  ```

--------------------------------------------------------------------------------

#### **2. GOVERNANÇA DE DADOS E VALIDAÇÃO ESTRITA (3-WAY SPLIT)**
* **Partição Estrita Pré-Modelagem:** Todo pipeline de dados DEVE ser particionado em 3 vias antes de qualquer transformação supervisionada ou treino de modelo:
  1. `D_train` (Treino Histórico): Utilizado para o `.fit()` dos transformadores do `@feature_engineering_specialist` e estimativa de parâmetros do modelo.
  2. `D_test` (Teste / Calibração Interna): Utilizado para busca de hiperparâmetros (Optuna), calibração de probabilidade (Isotonic/Platt) e otimização do corte financeiro ($c^*$).
  3. `D_val_oot` (Validação Out-Of-Time): Conjunto temporalmente isolado (dados mais recentes do histórico) mantido totalmente intocado até a auditoria final de estabilidade e ROI financeiro.
* **Isolamento de Target Leakage:** Todas as estatísticas de transformação (WOE, Target Encoding OOF, Imputação, Scalers) DEVEM executar `.fit()` estritamente em `D_train` e apenas `.transform()` em `D_test` e `D_val_oot`.

--------------------------------------------------------------------------------

#### **3. PROTOCOLO DE BENCHMARK PROGRESSIVO (3 ESTÁGIOS)**
Cada subagente de modelagem (`@classification_specialist`, `@logit_specialist`, `@causal_specialist`, `@linear_regression_wooldridge`) DEVE obrigatoriamente executar e reportar a matriz comparativa avaliada no conjunto `D_val_oot`:
* **Estágio 1 (Vanilla Model - Baseline):** Modelo simples (ex: Regressão Logística ou Árvore de Decisão rasa) com tratamento mínimo e imputação básica, estabelecendo a performance piso.
* **Estágio 2 (Modelo Robusto - Capacity Lift):** Algoritmo avançado (ex: CatBoost, XGBoost, GLM ElasticNet) com tuning de hiperparâmetros via Optuna sobre as variáveis brutas limpas, isolando o ganho da capacidade matemática do modelo ($\Delta_{\text{Capacity}}$).
* **Estágio 3 (FE Model - Feature Lift):** O mesmo algoritmo robusto alimentado pela matriz enriquecida pelo `@feature_engineering_specialist` (rolling windows, OOF Target Encoding, WoE, interações), isolando o ganho da engenharia de atributos ($\Delta_{\text{FE}}$).

--------------------------------------------------------------------------------

#### **4. ROTEAMENTO & WORKFLOW MULTIAGENTES (DAG)**
O fluxo de trabalho entre agentes no OpenCode segue um Grafo Dirigido Acíclico (DAG) estrito e sequencial:
1. `@orchestrator`: Realiza o Discovery dos 5 Pilares, emite o Scoping Charter, aprova as Unit Economics e impõe o Split em 3 Vias.
2. `@eda_specialist`: Executa a auditoria de integridade, testes de hipóteses bivariados e triagem de IV/Leakage estritamente em `D_train`.
3. `@feature_engineering_specialist`: Executa as transformações matemáticas, aprendendo parâmetros em `D_train` e aplicando `.transform()` em `D_test` e `D_val_oot`.
4. **Subagente de Modelagem:** Executa a esteira de benchmark em 3 estágios (Vanilla ➔ Robusto ➔ FE Model) e avalia nos 3 Objetivos Corporativos (Decidir $c^*$, Ordenar Top-$K$, Estimar Brier Score).
5. **Entrega Executiva:** Consolidação do relatório de ROI financeiro (em R$), explicabilidade SHAP em 2 níveis e decisão final de deploy.

--------------------------------------------------------------------------------

#### **5. ESTRUTURA DO PROJETO & CONVENÇÕES OPENCODE**
* **Localização de Subagentes:** Subagentes especializados devem residir exclusivamente no diretório `.opencode/agents/`.
* **Entrega de Artefatos:** Trabalhos de modelagem e análise devem ser salvos diretamente na pasta `notebooks/`.
* **Visualização de Dados & Gráficos (Seaborn):**
  * Biblioteca padrão: `seaborn` para gráficos estatísticos e dashboards executivos.
  * Tema padrão: Toda inicialização de script DEVE configurar o tema visual via `sns.set_theme(style="whitegrid")`.
  * Rótulos e eixos limpos, sem emojis, com paletas sóbrias para apresentação executiva.
* **Formatação de Markdown & Relatórios Executivos:**
  * Hierarquia de Títulos: Para seções principais e blocos executivos, utilize o padrão `#### **Título da Seção**`.
  * Destaques de Negócios: Destaque em **negrito** todas as métricas financeiras e estatísticas (**R$**, **ROI**, **KS**, **AUC**, **Brier Score**, **p-valor**).
  * **Proibição Estrita de Emojis:** Proibição do uso de emojis em títulos de seções, textos de markdown, logs de saída do terminal e comentários de código.
