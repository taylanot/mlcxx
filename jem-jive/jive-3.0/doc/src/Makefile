
.PHONY:			all install clean clean-all

.SUFFIXES:

.SUFFIXES:		.pdf .ps .tex .dvi .aux


TEXFILES 		= user-manual.tex \
			  include/definitions.tex \
			  chapter-1/chapter-1.tex \
			  chapter-2/chapter-2.tex \
			  chapter-3/chapter-3.tex \


all:			user-manual.pdf

install:		all
			cp -f user-manual.pdf ../

clean:
			@$(RM) *.dvi *.log *.toc *.aux *.out
			@$(RM) include/*.aux include/*.log

clean-all:		clean
			@$(RM) *.ps *.pdf


user-manual.dvi:	$(TEXFILES)


%.pdf:			%.ps
			ps2pdf $<

%.ps:			%.dvi
			dvips -t a4 $<

%.dvi:			%.tex
			latex $<
			latex $<

