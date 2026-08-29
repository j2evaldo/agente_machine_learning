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
- **Toda análise exploratória (EDA) e modelagem deve ser entregue em formato Jupyter Notebook (`.ipynb`) dentro do diretório `notebooks/`** (ex: `notebooks/01_eda_telco_churn.ipynb`).
- **Proibido gerar múltiplos arquivos soltos** (.csv ou .png avulsos de gráficos) a menos que explicitamente solicitado.
- **Gráficos e tabelas devem ser renderizados inline** nas células do próprio notebook.
- Os notebooks devem ser autodocumentados, alternando células de Markdown executivo (diagnósticos, hipóteses, conclusões) com blocos de código modular importando de `src/skills/` e `src/models/`.