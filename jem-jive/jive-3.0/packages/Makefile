
ifeq ($(origin JEMDIR),undefined)
  $(error undefined variable: JEMDIR)
endif

JIVEPATH	:= ..

include $(JIVEPATH)/makefiles/jive.mk
include $(JIVEPATH)/makefiles/private.mk


.PHONY:		all install clean lib lib-opt lib-dbg doc \
		examples test clean-doc clean-all


all:
		+@$(MakeSubdirs) -t all

install:
		+@$(MakeSubdirs) -t install

clean:
		+@$(MakeSubdirs) -t clean

lib:
		+@$(MakeSubdirs) -t lib

lib-opt:
		+@$(MakeSubdirs) -t lib-opt

lib-dbg:
		+@$(MakeSubdirs) -t lib-dbg


# Make doc twice to resolve circular references.

doc:
		+@$(MakeSubdirs) -t doc
		+@$(MakeSubdirs) -t doc

examples:
		+@$(MakeSubdirs) -t examples

test:
		+@$(MakeSubdirs) -t test

clean-doc:
		+@$(MakeSubdirs) -t clean-doc

clean-all:
		+@$(MakeSubdirs) -t clean-all
