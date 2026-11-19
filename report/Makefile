# Run from the repository root with: make -C report
.PHONY: all clean

all:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	@if grep -q '\\citation{' main.aux; then bibtex main; fi
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.synctex.gz main.toc main.pdf
