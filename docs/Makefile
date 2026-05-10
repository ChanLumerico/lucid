SPHINXOPTS  ?=
SPHINXBUILD  = sphinx-build
SOURCEDIR    = source
BUILDDIR     = _build

.PHONY: help html livehtml clean linkcheck

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo ""
	@echo "Build finished: file://$(abspath $(BUILDDIR)/html/index.html)"

livehtml:
	sphinx-autobuild \
	    $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS) \
	    --ignore "*.pyc" \
	    --ignore "_build" \
	    --watch ../../lucid \
	    --open-browser

linkcheck:
	@$(SPHINXBUILD) -b linkcheck $(SOURCEDIR) $(BUILDDIR)/linkcheck $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
	@echo "Build directory cleaned."
