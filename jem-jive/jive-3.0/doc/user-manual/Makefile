

PDWEAVE_PATH =		examples


TEX_SOURCES = 		definitions.tex \
			user-manual.tex

PD_SOURCES =		chapter-1.pd \
			chapter-2.pd \
			chapter-3.pd \
			chapter-4.pd


.PHONY:			dvi ps manual.tex



dvi:			user-manual.dvi


ps:			user-manual.ps


user-manual.dvi:	manual.tex
			latex user-manual.tex

user-manual.ps:		user-manual.dvi
			dvips user-manual


manual.tex:		$(TEX_SOURCES) $(PD_SOURCES:.pd=.tex)


clean:
			$(RM) *.aux *.toc *.log

clean-all:		clean
			$(RM) *.dvi *.ps
			$(RM) $(PD_SOURCES:.pd=.tex)
			$(RM) _pd*



%.tex : %.pd
			export PDWEAVE_PATH="$(PDWEAVE_PATH)"; \
			pdweave --center --linenr --frame --listing $<
