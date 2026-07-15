# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

NOTEBOOK_SOURCES = estimands_toy_game.ipynb masking_views.ipynb runtime_core.ipynb exact_estimation.ipynb estimator_comparison.ipynb attribution_viz.ipynb text_explain_e2e.ipynb

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile copy-notebooks render-guide-outputs

copy-notebooks:
	@mkdir -p notebooks
	@for nb in $(NOTEBOOK_SOURCES); do \
		cp ../examples/$$nb notebooks/$$nb; \
	done

render-guide-outputs:
	@cd .. && uv run python docs/scripts/render_guide_outputs.py

%: Makefile copy-notebooks render-guide-outputs
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
