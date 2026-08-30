# Diretrizes Globais do Projeto (AGENTS.md)

## 1. Ambiente e Ferramentas
- Use estritamente o gerenciador `uv` para executar qualquer comando, script ou suite de testes.
- Rodar testes: `uv run pytest -v`
- Linting e formatação: `uv run ruff check .`
- Nunca instale pacotes no sistema sem registrá-los no `pyproject.toml`.

## 2. Padrões de Repositório e Commits
- Todos os commits devem seguir a convenção de Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`).
- Código executável e modular deve ficar em `src/`, testes em `tests/` e dados brutos em `data/raw/` (somente leitura).

## 3. Governança e Transição de Fases (CRISP-DM)
- O `@orchestrator` deve conduzir o Discovery inicial e obter a aprovação do Scoping Charter antes de delegar qualquer tarefa de modelagem.
- O `@eda_specialist` (ou as skills em `src/skills/eda_tools.py`) deve auditar nulos, leakage e WoE/IV antes da estimação de modelos.
- `@logit_specialist` e `@linear_regression_wooldridge` devem sempre traduzir parâmetros em impactos de negócio e métricas financeiras.

## 4. Padrões de Artefatos e Entregas (Notebooks)
- Toda análise exploratória (EDA) e modelagem deve ser entregue em formato Jupyter Notebook (`.ipynb`) dentro do diretório `notebooks/`.
- Proibido gerar múltiplos arquivos soltos (.csv ou .png avulsos de gráficos). Tabelas e gráficos devem ser renderizados inline no notebook.
- Os notebooks devem alternar células de Markdown estruturado com blocos de código executável.
- **Padronização Gráfica:** Todos os scripts e notebooks devem configurar obrigatoriamente o tema visual do Seaborn logo na inicialização (`import seaborn as sns; sns.set_theme(style="darkgrid")` ou estilo equivalente `plt.style.use('seaborn-v0_8-darkgrid')`), garantindo fundo cinza com grid branco e contraste estruturado.

## 5. Diretrizes de Conteúdo e Insights de Negócio
- Ao final de cada análise bivariada, adicione uma célula de Markdown com "Insight de Negócio", traduzindo o resultado estatístico (p-valor, Information Value, separação de classes) em implicações práticas para a operação.
- Todo notebook deve obrigatoriamente terminar com uma seção intitulada "Resumo Executivo", consolidando:
  1. Principais achados empíricos e fatores de risco.
  2. Ranking de variáveis mais preditivas/relevantes.
  3. Recomendações acionáveis para as áreas de negócio e próximos passos analíticos.

## 6. Restrições Estritas de Formatação
- Proibido o uso de qualquer tipo de emoji em títulos, textos de Markdown, gráficos, saídas de terminal, relatórios executivos ou comentários de código. A linguagem deve ser 100% formal, técnica e limpa.

## 7. Padrões de Modelagem, Persistência e Recomendações de Negócio
- Todo processo de modelagem deve salvar a base final com predições/escores em `data/processed/[nome_projeto]_scored.csv`.
- Modelos serializados e matrizes de parâmetros devem ser salvos em `models/`.
- **Uso de Skills Determinísticas:** É obrigatório utilizar as funções de `src/skills/decision_tools.py` e `src/skills/eda_tools.py` para auditorias, curvas de lucro e simulações de capacidade, evitando loops manuais nos notebooks.
- **Calibração e Capacidade:** Modelos voltados para decisão financeira devem:
  1. Utilizar `class_weight=None` no Scikit-Learn para manter equivalência com Statsmodels
  2. Realizar análises de capacidade na base completa (n_total), não apenas na partição de teste
  3. Documentar justificativa para a estratégia de threshold utilizada (c* teórico vs Top-K)
- **Seção de Fechamento:** Os notebooks de modelagem devem conter obrigatoriamente uma seção final intitulada "Resumo Executivo", estruturada por audiência:
  1. **Decisão Financeira Ótima** (destaque principal para executivos): Limiar de probabilidade ótimo (c*) e resultado financeiro líquido projetado em R$.
  2. **Alavancas Práticas** (ação imediata para o time de negócio): Quem atacar (perfil de risco), o que oferecer (bundle, desconto, fidelização) e métricas de impacto em p.p.
  3. **Diagnóstico Técnico** (seção à parte para validação do modelo): Métricas de ajuste (Pseudo R2, AUC, Brier Score) e testes de premissas — conteúdo opcional para executivos, obrigatório para auditória técnica.