ifndef JIVE_MK
       JIVE_MK = 1

#########################################################################
# JEMDIR & JEMPATH
#########################################################################


ifndef JEMPATH
  ifndef JEMDIR
    $(error undefined variable: JEMDIR)
  endif
  JEMPATH  := $(JEMDIR)
endif

# JIVEPATH overrides JIVEDIR

ifdef JIVEPATH
  JIVEDIR  := $(shell cd $(JIVEPATH); pwd)
else
  JIVEPATH := $(JIVEDIR)
endif


export JIVEDIR


#########################################################################
# Include files
#########################################################################


include $(JEMPATH)/makefiles/packages/base.mk
include $(JEMPATH)/makefiles/packages/io.mk
include $(JEMPATH)/makefiles/packages/util.mk
include $(JEMPATH)/makefiles/packages/mp.mk
include $(JEMPATH)/makefiles/packages/numeric.mk
include $(JEMPATH)/makefiles/packages/xml.mk
include $(JEMPATH)/makefiles/packages/xutil.mk


#########################################################################
# Variables & Constants
#########################################################################


JIVE_PACKAGES    =

JIVE_LIBS        =
JIVE_LIBDIRS     = $(JIVEPATH)/lib
JIVE_LIBDIRS_OPT = $(JIVEPATH)/lib-opt $(JIVE_LIBDIRS)
JIVE_LIBDIRS_DBG = $(JIVEPATH)/lib-dbg $(JIVE_LIBDIRS)
JIVE_INCDIRS     = $(JIVEPATH)/include

LIBS             = $(MY_LIBS) $(JIVE_LIBS) $(JEM_LIBS) $(SYS_LIBS)
LIBDIRS          = $(MY_LIBDIRS) $(JIVE_LIBDIRS) $(JEM_LIBDIRS)\
		   $(SYS_LIBDIRS)
LIBDIRS_OPT      = $(MY_LIBDIRS_OPT) $(JIVE_LIBDIRS_OPT)\
		   $(JEM_LIBDIRS_OPT) $(SYS_LIBDIRS)
LIBDIRS_DBG      = $(MY_LIBDIRS_DBG) $(JIVE_LIBDIRS_DBG)\
		   $(JEM_LIBDIRS_DBG) $(SYS_LIBDIRS)
INCDIRS          = $(MY_INCDIRS) $(JIVE_INCDIRS) $(JEM_INCDIRS)\
		   $(SYS_INCDIRS)


endif
