# HYGIA — Graph RAG para segurança medicamentosa
# Ambiente isolado via uv; alvos reprodutíveis ponta a ponta.

PY := .venv/bin/python
UV := uv

.PHONY: help setup pipeline test app figuras smoke clean

help:
	@echo "Alvos disponíveis:"
	@echo "  make setup     - cria .venv e instala dependências pinadas (uv)"
	@echo "  make pipeline  - executa o pipeline completo (semente -> resultados + figuras)"
	@echo "  make figuras   - regenera apenas as figuras a partir dos resultados"
	@echo "  make test      - roda a suíte de testes (smoke-test do CI)"
	@echo "  make smoke     - pipeline rápido, sem figuras (verificação de fumaça)"
	@echo "  make app       - inicia a interface Streamlit"
	@echo "  make clean     - remove artefatos gerados"

setup:
	$(UV) venv .venv --python 3.12
	$(UV) pip install -p $(PY) -r requirements.txt

pipeline:
	PYTHONPATH=src $(PY) scripts/pipeline.py

smoke:
	PYTHONPATH=src $(PY) scripts/pipeline.py --rapido

figuras:
	PYTHONPATH=src $(PY) scripts/figuras.py

test:
	PYTHONPATH=src $(PY) -m pytest tests/ -q

app:
	PYTHONPATH=src $(PY) -m streamlit run app/streamlit_app.py

clean:
	rm -rf data/corpus/* data/processed/* resultados/*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
