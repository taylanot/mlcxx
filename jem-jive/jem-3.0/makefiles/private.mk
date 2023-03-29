
InstallPackage  = $(JEMPATH)/scripts/bin/install-package

MakeLib         = $(JEMPATH)/scripts/bin/make-package-lib\
		  --libpath=$(JEMPATH)/lib $(OBJDIR)

MakeLibOpt      = $(JEMPATH)/scripts/bin/make-package-lib\
		  --libpath=$(JEMPATH)/lib-opt $(OBJDIR_OPT)

MakeLibDbg      = $(JEMPATH)/scripts/bin/make-package-lib\
		  --libpath=$(JEMPATH)/lib-dbg $(OBJDIR_DBG)

CXX_FLAGS       = $(CXX_STD_FLAGS) $(CXX_DLL_FLAGS)\
		  $(JEM_CXX_BLD_FLAGS)

