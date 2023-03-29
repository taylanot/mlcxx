
.PHONY:		all install clean doc clean-doc clean-all


all:
		+@$(MakeSubdirs) -t all src

install:
		+@$(MakeSubdirs) -t install doc

clean:
		+@$(MakeSubdirs) -t clean src

doc:
		+@$(MakeSubdirs) doc

clean-doc:
		+@$(MakeSubdirs) -t clean doc

clean-all:	clean clean-doc

