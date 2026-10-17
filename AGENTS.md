# Diretrizes Globais do Projeto (AGENTS.md)

Fonte unica de governanca do repositorio. Documento carregado automaticamente pelo opencode.

## 1. Ambiente e Ferramentas
- Use estritamente o gerenciador `uv` para executar qualquer comando, script ou suite de testes.
- Python obrigatorio: **3.12** (pin em `.python-version`). Proibido o uso de 3.10/3.11 ou Python global do sistema.
- Comandos padrao:
  - Execucao isolada de script: `uv run python <script.py>`
  - Adicao de dependencia: `uv add <pacote>`
  - Testes: `uv run pytest -v`
  - Linting e formatacao: `uv run ruff check .`
- Nunca instale pacotes no sistema sem registra-los no `pyproject.toml`.
- **PEP 723 Inline Script Metadata:** todo script Python autonomo gerado pelos agentes deve conter o cabecalho de dependencias para execucao reprodutivel via `uv run`:
  ```python
  # /// script
  # requires-python = ">=3.12"
  # dependencies = [
  #     "pandas>=2.2.0",
  #     "numpy>=1.26.0",
  #     "scikit-learn>=1.5.0",
  #     "catboost>=1.2.0",
  #     "optuna>=3.5.0",
  #     "shap>=0.44.0",
  #     "seaborn>=0.13.0",
  #     "statsmodels>=0.14.0",
  # ]
  # ///
  ```

## 2. Padroes de Repositorio e Commits
- Todos os commits devem seguir a convencao de Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`).
- Codigo executavel e modular fica em `src/`, testes em `tests/` e dados brutos em `data/raw/` (somente leitura).
- Estrutura base do repositorio:
  - `.opencode/agents/` — definicoes de subagentes (uma `.md` por agente).
  - `.opencode/commands/` — comandos customizados (opcional).
  - `.opencode/skills/` — skills nativas do opencode (`<nome>/SKILL.md`).
  - `src/skills/` — funcoes determinasticas reutilizaveis em Python.
  - `notebooks/`, `data/{raw,interim,processed}/`, `models/` — artefatos de projeto (conteudo ignorado no Git; estrutura versionada via `.gitkeep`).

## 3. Governanca e Transicao de Fases (CRISP-DM)
- O `@orchestrator` conduz o Discovery inicial e obtem a aprovacao do Scoping Charter antes de delegar qualquer tarefa de modelagem.
- O `@eda_specialist` (ou as skills em `src/skills/eda_tools.py`) audita nulos, leakage e WoE/IV antes da estimacao de modelos.
- `@logit_specialist` e `@linear_regression_wooldridge` devem sempre traduzir parametros em impactos de negocio e metricas financeiras.

### 3.1 Fluxo Multiagentes (DAG)
1. `@orchestrator`: Discovery dos 5 Pilares, emissao do Scoping Charter, aprovacao das Unit Economics e imposicao do Split em 3 Vias.
2. `@eda_specialist`: auditoria de integridade, testes de hipoteses bivariados e triagem de IV/Leakage estritamente em `D_train`.
3. `@feature_engineering_specialist`: transformacoes com `.fit()` em `D_train` e `.transform()` em `D_test` e `D_val_oot`.
4. Subagente de modelagem: benchmark em 3 estagios (Vanilla -> Robusto -> FE Model) e avaliacao nos 3 objetivos corporativos (Decidir `c*`, Ordenar Top-K, Estimar Brier Score).
5. Entrega executiva: consolidacao do ROI financeiro (R$), explicabilidade SHAP em 2 niveis e decisao final de deploy.

## 4. Particao de Dados em 3 Vias e Anti-Leakage
- Todo pipeline deve ser particionado em 3 vias antes de qualquer transformacao supervisionada ou treino:
  1. `D_train` (Treino): `.fit()` dos transformadores e estimativa dos parametros.
  2. `D_test` (Teste / Calibracao Interna): tuning (Optuna), calibracao de probabilidade e otimizacao do corte financeiro (`c*`).
  3. `D_val_oot` (Validacao Out-Of-Time): conjunto temporalmente isolado, mantido intocado ate a auditoria final de estabilidade e ROI.
- Todas as estatisticas de transformacao (WoE, Target Encoding OOF, imputacao, scalers) devem executar `.fit()` estritamente em `D_train` e apenas `.transform()` em `D_test` e `D_val_oot`.

## 5. Protocolo de Benchmark Progressivo (3 Estagios)
Cada subagente de modelagem deve executar e reportar a matriz comparativa avaliada em `D_val_oot`:
1. **Estagio 1 (Vanilla Model):** modelo simples com tratamento minimo, estabelecendo a performance piso.
2. **Estagio 2 (Modelo Robusto):** algoritmo avancado com tuning de hiperparametros sobre variaveis brutas limpas, isolando o ganho de capacidade (`Delta_Capacity`).
3. **Estagio 3 (FE Model):** o mesmo algoritmo robusto alimentado pela matriz enriquecida pelo `@feature_engineering_specialist`, isolando o ganho da engenharia de atributos (`Delta_FE`).

## 6. Padroes de Artefatos e Entregas (Notebooks)
- Toda analise exploratoria (EDA) e modelagem deve ser entregue em formato Jupyter Notebook (`.ipynb`) dentro do diretorio `notebooks/`.
- Proibido gerar multiplos arquivos soltos (.csv ou .png avulsos de graficos). Tabelas e graficos devem ser renderizados inline no notebook.
- Os notebooks devem alternar celulas de Markdown estruturado com blocos de codigo executavel.
- **Padronizacao Grafica:** todos os scripts e notebooks devem configurar o tema visual do Seaborn na inicializacao (`import seaborn as sns; sns.set_theme(style="darkgrid")` ou estilo equivalente `plt.style.use('seaborn-v0_8-darkgrid')`), garantindo fundo cinza com grid branco e contraste estruturado.

## 7. Diretrizes de Conteudo e Insights de Negocio
- Ao final de cada analise bivariada, adicione uma celula de Markdown com "Insight de Negocio", traduzindo o resultado estatistico (p-valor, Information Value, separacao de classes) em implicacoes praticas para a operacao.
- Todo notebook deve obrigatoriamente terminar com uma secao intitulada "Resumo Executivo", consolidando:
  1. Principais achados empiricos e fatores de risco.
  2. Ranking de variaveis mais preditivas/relevantes.
  3. Recomendacoes acionaveis para as areas de negocio e proximos passos analiticos.

## 8. Padroes de Modelagem, Persistencia e Recomendacoes de Negocio
- Todo processo de modelagem deve salvar a base final com predicoes/escores em `data/processed/[nome_projeto]_scored.csv`.
- Modelos serializados e matrizes de parametros devem ser salvos em `models/`.
- **Uso de Skills Deterministicas:** e obrigatorio utilizar as funcoes de `src/skills/decision_tools.py`, `src/skills/eda_tools.py` e `src/skills/recommender_tools.py` para auditorias, curvas de lucro e simulacoes de capacidade, evitando loops manuais nos notebooks.
- **Calibracao e Capacidade:** modelos voltados para decisao financeira devem:
  1. Utilizar `class_weight=None` no Scikit-Learn para manter equivalencia com Statsmodels.
  2. Realizar analises de capacidade na base completa (`n_total`), nao apenas na particao de teste.
  3. Documentar justificativa para a estrategia de threshold utilizada (c* teorico vs Top-K).
- **Secao de Fechamento:** os notebooks de modelagem devem conter obrigatoriamente uma secao final intitulada "Resumo Executivo", estruturada por audiencia:
  1. **Decisao Financeira Otima** (destaque principal para executivos): limiar de probabilidade otimo (c*) e resultado financeiro liquido projetado em R$.
  2. **Alavancas Praticas** (acao imediata para o time de negocio): quem atacar (perfil de risco), o que oferecer (bundle, desconto, fidelizacao) e metricas de impacto em p.p.
  3. **Diagnostico Tecnico** (secao a parte para validacao do modelo): metricas de ajuste (Pseudo R2, AUC, Brier Score) e testes de premissas — conteudo opcional para executivos, obrigatorio para auditoria tecnica.

## 9. Restricoes Estritas de Formatacao
- Proibido o uso de qualquer tipo de emoji em titulos, textos de Markdown, graficos, saidas de terminal, relatorios executivos ou comentarios de codigo. A linguagem deve ser 100% formal, tecnica e limpa.
- Hierarquia de titulos: para secoes principais e blocos executivos, utilize o padrao `#### **Titulo da Secao**`.
- Destaques de negocios: destaque em **negrito** todas as metricas financeiras e estatisticas (**R$**, **ROI**, **KS**, **AUC**, **Brier Score**, **p-valor**).
