
export JEMDIR   = $(CURDIR)

include makefiles/jem.mk
include makefiles/private.mk

CONFIG_FILE	= scripts/include/config.sh

configured      = $(shell test -e $(CONFIG_FILE) && echo 1)


.PHONY:		all install clean lib lib-opt lib-dbg debug doc\
		examples test clean-doc clean-all distclean\
		clean-source


all:		$(CONFIG_FILE)
		@$(MAKE) -C packages all

install:	$(CONFIG_FILE)
		@$(MAKE) -C packages install

lib:		$(CONFIG_FILE)
		@$(MAKE) -C packages lib

lib-opt:	$(CONFIG_FILE)
		@$(MAKE) -C packages lib-opt

lib-dbg:	$(CONFIG_FILE)
		@$(MAKE) -C packages lib-dbg

debug:		lib-dbg

doc:		$(CONFIG_FILE)
		@scripts/bin/make-doc

examples:	$(CONFIG_FILE)
		@$(MAKE) -C packages examples

test:		$(CONFIG_FILE)
		@$(MAKE) -C packages test


ifneq ($(configured),1)

clean:

clean-doc:

clean-all:

distclean:

else

clean:
		@$(MAKE) -C packages clean

clean-doc:
		@$(RM_R) doc/doxygen_sqlite3.db doc/html

clean-all:
		@$(MAKE) -C packages clean-all
		@if [ -d lib ];     then $(RM) lib/*;     fi
		@if [ -d lib-opt ]; then $(RM) lib-opt/*; fi
		@if [ -d lib-dbg ]; then $(RM) lib-dbg/*; fi

distclean:	clean-all
		@if [ -e configure.log ]; then \
		  $(RM) configure.log; \
		fi
		@scripts/bin/unconfigure

endif


clean-source:	$(CONFIG_FILE)
		@if [ -x scripts/bin/clean-source ]; then \
		  scripts/bin/clean-source packages ; \
		fi


$(CONFIG_FILE):
		./configure
