# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

NOTEBOOK_SOURCES = estimands_toy_game.ipynb masking_views.ipynb runtime_core.ipynb exact_estimation.ipynb

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile copy-notebooks

copy-notebooks:
	@mkdir -p notebooks
	@for nb in $(NOTEBOOK_SOURCES); do \
		cp ../examples/$$nb notebooks/$$nb; \
	done

%: Makefile copy-notebooks
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
