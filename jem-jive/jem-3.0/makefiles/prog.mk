
ifndef JEMPATH
  JEMPATH := $(JEMDIR)
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

program			:= $(strip $(basename $(program)))

ifeq ($(program),)
  $(error no name for the program has been specified)
endif

program-opt		:= $(program)$(EXE_SUFFIX_OPT)
program-dbg		:= $(program)$(EXE_SUFFIX_DBG)
program			:= $(program)$(EXE_SUFFIX)

ifndef objects

  objects		:= $(wildcard *.cpp)

  ifdef subdirs

    objects		+= $(foreach dir,$(subdirs),\
				$(wildcard $(dir)/*.cpp))

    VPATH		 = $(subdirs)

  endif

endif

objects			:= $(notdir $(objects))
objects			:= $(basename $(objects))
objects			:= $(addsuffix $(OBJ_SUFFIX),$(objects))
objects-opt		:= $(addprefix $(OBJBASE_OPT),$(objects))
objects-dbg		:= $(addprefix $(OBJBASE_DBG),$(objects))
objects			:= $(addprefix $(OBJBASE),$(objects))
objects-dbg		:= $(objects-dbg) $(DBG_TOOLS)


#########################################################################
# Rules
#########################################################################


.SUFFIXES:

.PHONY:			all opt debug install clean clean-all

all:			$(program)

opt:			$(program-opt)

debug:			$(program-dbg)

install:		$(program)
			strip $(program)

clean:
			@$(CleanObjFiles)
			@$(RM) $(CORE_DUMPS)

clean-all:		clean
			$(RM) $(program) $(program-opt) $(program-dbg)


$(program):		$(objects)
			$(LinkExe)

$(program-opt):		$(objects-opt)
			$(LinkExe)

$(program-dbg):		$(objects-dbg)
			$(LinkExe)

$(program-opt):		CMODE = $(CMODE_OPT)

$(program-dbg):		CMODE = $(CMODE_DBG)


include			$(JEMPATH)/makefiles/obj.mk
