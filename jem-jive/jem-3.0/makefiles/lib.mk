
ifndef JEMPATH
  JEMPATH = $(JEMDIR)
endif

include $(JEMPATH)/makefiles/jem.mk


#########################################################################
# Variables
#########################################################################


ifndef OBJDIR
  OBJDIR		:= OBJ
endif

OBJDIR_OPT		:= $(OBJDIR)_OPT
OBJDIR_DBG		:= $(OBJDIR)_DBG

OBJBASE			:= $(OBJDIR)/$(OBJ_PREFIX)
OBJBASE_OPT		:= $(OBJDIR_OPT)/$(OBJ_PREFIX)
OBJBASE_DBG		:= $(OBJDIR_DBG)/$(OBJ_PREFIX)

library			:= $(strip $(basename $(library)))

ifeq ($(library),)
  $(error no name for the shared library has been specified)
endif

library-opt		:= $(LIB_PREFIX)$(library)-opt$(LIB_SUFFIX)
library-dbg		:= $(LIB_PREFIX)$(library)-dbg$(LIB_SUFFIX)
library			:= $(LIB_PREFIX)$(library)$(LIB_SUFFIX)

ifndef objects
  objects		:= $(wildcard *.cpp)
endif

objects			:= $(basename $(objects))
objects			:= $(addsuffix $(OBJ_SUFFIX),$(objects))
objects-opt		:= $(addprefix $(OBJBASE_OPT),$(objects))
objects-dbg		:= $(addprefix $(OBJBASE_DBG),$(objects))
objects			:= $(addprefix $(OBJBASE),$(objects))
objects-dbg		:= $(objects-dbg) $(DBG_TOOLS)

CXX_FLAGS		 = $(CXX_STD_FLAGS) $(CXX_LIB_FLAGS)


#########################################################################
# Rules
#########################################################################


.SUFFIXES:

.PHONY:			all opt debug clean clean-all

all:			$(library)

opt:			$(library-opt)

debug:			$(library-dbg)

install:		$(library)

clean:
			@$(CleanObjFiles)
			@$(RM) $(CORE_DUMPS)

clean-all:		clean
			$(RM) $(library) $(library-opt) $(library-dbg)


$(library):		$(objects)
			$(LinkLib)

$(library-opt):		$(objects-opt)
			$(LinkLib)

$(library-dbg):		$(objects-dbg)
			$(LinkLib)

$(library-opt):		CMODE = $(CMODE_OPT)

$(library-dbg):		CMODE = $(CMODE_DBG)


include			$(JEMPATH)/makefiles/obj.mk
