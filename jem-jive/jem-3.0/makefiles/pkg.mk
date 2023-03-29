
.PHONY:         all install clean lib lib-opt lib-dbg \
		examples test clean-all


all:
		+@$(MakeSubdirs) -t all src debug

install:
		@$(InstallPackage)

clean:
		@$(RM) configure.log
		+@$(MakeSubdirs) -t clean src debug test examples

lib:
		+@$(MakeSubdirs) -t lib src debug

lib-opt:
		+@$(MakeSubdirs) -t lib-opt src

lib-dbg:
		+@$(MakeSubdirs) -t lib-dbg src

examples:
		+@$(MakeSubdirs) examples

test:
		+@$(MakeSubdirs) test

clean-all:	clean


