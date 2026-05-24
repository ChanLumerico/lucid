# =====================================================================
# Lucid Tech Report --- Makefile
#
# Targets:
#   make            Incremental build (latexmk handles dependency tracking)
#   make watch      Continuous build, rebuild on file changes
#   make once       Single-pass build (no auxiliary rounds, for debugging)
#   make clean      Remove aux files, keep PDF
#   make realclean  Remove everything, including PDF
#   make count      Quick word count of all .tex sources
# =====================================================================

LATEXMK := latexmk
MAIN    := main
PDF     := $(MAIN).pdf

.PHONY: all watch once clean realclean count help

all: $(PDF)

$(PDF): $(MAIN).tex
	$(LATEXMK) $(MAIN)

watch:
	$(LATEXMK) -pvc $(MAIN)

once:
	xelatex -shell-escape -interaction=nonstopmode -file-line-error $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN)

realclean:
	$(LATEXMK) -C $(MAIN)
	rm -f $(MAIN).bbl $(MAIN).run.xml

count:
	@find frontmatter parts chapters appendices -name '*.tex' \
		-exec wc -w {} + | tail -1

help:
	@echo "Targets: all watch once clean realclean count"
