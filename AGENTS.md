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
- Os notebooks de modelagem devem conter obrigatoriamente uma seção final intitulada "Plano de Ação e Recomendações de Negócio", detalhando:
  1. Decisão Ótima: Limiar de probabilidade ótimo (c*) ou Preço ótimo (P*) com impacto financeiro estimado em R$.
  2. Alavancas Práticas: Ações recomendadas para as equipes de negócio baseadas nos coeficientes e efeitos marginais (AME).
  3. Matriz de Risco e Trade-offs: Custos de oportunidade e limites da capacidade operacional.
  4. Sempre traga receomdações claras para o time de negócios utilizando linguagem não técnica para que tods possam entender e aplicar em suas rotinas de trabalho