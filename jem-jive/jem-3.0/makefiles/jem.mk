ifndef JEM_MK
       JEM_MK = 1

#########################################################################
# JEMDIR & JEMPATH
#########################################################################

# JEMPATH overrides JEMDIR

ifdef JEMPATH
  JEMDIR  := $(shell cd $(JEMPATH); pwd -P)
else
  JEMPATH := $(JEMDIR)
endif


#########################################################################
# Include files
#########################################################################


 include $(JEMPATH)/makefiles/defaults.mk
-include $(JEMPATH)/makefiles/config.mk

ifdef SYS_NAME

  -include $(JEMPATH)/makefiles/config-$(SYS_NAME).mk

  ifdef CXX_NAME
    -include $(JEMPATH)/makefiles/config-$(SYS_NAME)-$(CXX_NAME).mk
  endif

endif


#########################################################################
# Variables & Constants
#########################################################################


JEM_PACKAGES		=

JEM_LIBDIRS		= $(JEMPATH)/lib
JEM_LIBDIRS_OPT		= $(JEMPATH)/lib-opt $(JEM_LIBDIRS)
JEM_LIBDIRS_DBG		= $(JEMPATH)/lib-dbg $(JEM_LIBDIRS)
JEM_INCDIRS		= $(JEMPATH)/include

LIBS			= $(MY_LIBS) $(JEM_LIBS) $(SYS_LIBS)
LIBDIRS			= $(MY_LIBDIRS) $(JEM_LIBDIRS)\
			  $(SYS_LIBDIRS)
LIBDIRS_OPT		= $(MY_LIBDIRS_OPT) $(JEM_LIBDIRS_OPT)\
			  $(SYS_LIBDIRS)
LIBDIRS_DBG		= $(MY_LIBDIRS_DBG) $(JEM_LIBDIRS_DBG)\
			  $(SYS_LIBDIRS)
INCDIRS			= $(MY_INCDIRS) $(JEM_INCDIRS) $(SYS_INCDIRS)

CXX_LNK_FLAGS		= $(MY_CXX_LNK_FLAGS) $(JEM_CXX_LNK_FLAGS)
CXX_STD_FLAGS		= $(MY_CXX_STD_FLAGS) $(JEM_CXX_STD_FLAGS)
CXX_DLL_FLAGS		= $(MY_CXX_DLL_FLAGS) $(JEM_CXX_DLL_FLAGS)
CXX_LIB_FLAGS		= $(MY_CXX_LIB_FLAGS) $(JEM_CXX_LIB_FLAGS)
CXX_DBG_FLAGS		= $(MY_CXX_DBG_FLAGS) $(JEM_CXX_DBG_FLAGS)
CXX_PRF_FLAGS		= $(MY_CXX_PRF_FLAGS) $(JEM_CXX_PRF_FLAGS)
CXX_OPT_FLAGS1		= $(MY_CXX_OPT_FLAGS1) $(JEM_CXX_OPT_FLAGS1)
CXX_OPT_FLAGS2		= $(MY_CXX_OPT_FLAGS2) $(JEM_CXX_OPT_FLAGS2)
CXX_OPT_FLAGS3		= $(MY_CXX_OPT_FLAGS3) $(JEM_CXX_OPT_FLAGS3)

CXX_FLAGS        	= $(CXX_STD_FLAGS)
CXX_FLAGS_OPT    	= $(CXX_FLAGS) $(CXX_OPT_FLAGS$(OPT_LEVEL))
CXX_FLAGS_DBG    	= $(CXX_FLAGS) $(CXX_DBG_FLAGS)

MY_LIBDIRS_OPT   	= $(MY_LIBDIRS)
MY_LIBDIRS_DBG   	= $(MY_LIBDIRS)
MY_CXX_OPT_FLAGS1	= $(MY_CXX_OPT_FLAGS)
MY_CXX_OPT_FLAGS2	= $(MY_CXX_OPT_FLAGS)
MY_CXX_OPT_FLAGS3	= $(MY_CXX_OPT_FLAGS)

CMODE			=
CMODE_OPT		= _OPT
CMODE_DBG		= _DBG


#########################################################################
# Exported variables
#########################################################################


export			MAKE
export			JEMDIR


#########################################################################
# Commands
#########################################################################


cxx			= @$(JEMPATH)/bin/CC -f-
cxx-flags		= -f'$(subst ','\'',$(CXX_FLAGS$(CMODE)))'
cxx-lnk-flags		= -f'$(subst ','\'',$(CXX_LNK_FLAGS))'
cxx-libs		= -l'$(subst ','\'',$(LIBS))'
cxx-libdirs		= -L'$(subst ','\'',$(LIBDIRS$(CMODE)))'
cxx-incdirs		= -I'$(subst ','\'',$(INCDIRS))'

Compile			= $(cxx) -c $(cxx-flags) $(cxx-incdirs)\
			  -o $@ -- $<

CompileExe		= $(cxx) $(cxx-flags) $(cxx-lnk-flags)\
			  $(cxx-incdirs) $(cxx-libdirs) $(cxx-libs)\
			  -o $@ -- $^

LinkExe			= $(cxx) $(cxx-flags) $(cxx-lnk-flags)\
			  $(cxx-libdirs) $(cxx-libs)\
			  -o $@ -- $^

LinkDLL			= $(cxx) $(cxx-flags) $(cxx-lnk-flags)\
			  $(cxx-libdirs) $(cxx-libs)\
			  -shared -o $@ -- $^

LinkLib 		= @$(JEMPATH)/bin/mklib $@ $^

MakeSubdirs		= $(JEMPATH)/bin/submake

ListSubdirs		= $(shell \
		              for d in * ; do \
			        if [ -e $$d/DISABLED ]; then \
                                  continue ; \
                                fi ; \
                                if [ -e $$d/Makefile ] ; then \
                                  echo $$d ; \
                                fi ; \
                              done )

endif
