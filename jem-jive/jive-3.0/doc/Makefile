
JIVEPATH	:= ..

include $(JIVEPATH)/makefiles/jive.mk
include $(JIVEPATH)/makefiles/private.mk

all:
		$(JIVEDIR)/scripts/make-main-index
		latex user-manual
		latex user-manual
		dvips user-manual

clean:
		@$(RM) *.dvi *.log *.toc *.aux


clean-all:	clean
		@$(RM_R) doxygen_sqlite3.db html

