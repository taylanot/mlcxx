
InstallPackage  = $(JIVEPATH)/scripts/bin/install-package

MakeLib         = $(JIVEPATH)/scripts/bin/make-package-lib\
		  --libpath=$(JIVEPATH)/lib $(OBJDIR)

MakeLibOpt      = $(JIVEPATH)/scripts/bin/make-package-lib\
		  --libpath=$(JIVEPATH)/lib-opt $(OBJDIR_OPT)

MakeLibDbg      = $(JIVEPATH)/scripts/bin/make-package-lib\
		  --libpath=$(JIVEPATH)/lib-dbg $(OBJDIR_DBG)


CXX_FLAGS       = $(CXX_STD_FLAGS) $(CXX_DLL_FLAGS)\
		  $(JEM_CXX_BLD_FLAGS)
