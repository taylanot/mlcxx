
ifndef JEM_MK
  $(error the makefile $$(JEMDIR)/makefiles/jem.mk\
          has not been included)
endif

ifndef JEMPATH
  JEMPATH = $(JEMDIR)
endif


#########################################################################
# Variables
#########################################################################


ifndef OBJDIR
  OBJDIR	:= OBJ
endif

OBJDIR_OPT	:= $(OBJDIR)_OPT
OBJDIR_DBG	:= $(OBJDIR)_DBG

ifndef OBJ_PREFIX
  OBJ_PREFIX    := $(shell $(JEMPATH)/bin/objprefix -p)
endif

OBJBASE		:= $(OBJDIR)/$(OBJ_PREFIX)
OBJBASE_OPT	:= $(OBJDIR_OPT)/$(OBJ_PREFIX)
OBJBASE_DBG	:= $(OBJDIR_DBG)/$(OBJ_PREFIX)

ifndef objects
  objects	:= $(wildcard *.cpp)
endif

objects		:= $(basename $(objects))
objects		:= $(addsuffix $(OBJ_SUFFIX),$(objects))
objects-opt	:= $(addprefix $(OBJBASE_OPT),$(objects))
objects-dbg	:= $(addprefix $(OBJBASE_DBG),$(objects))
objects		:= $(addprefix $(OBJBASE),$(objects))

SUBDIRS		:= $(ListSubdirs)

#########################################################################
# Rules
#########################################################################


.SUFFIXES:

.PHONY:		all install clean lib lib-opt lib-dbg \
		subdirs $(SUBDIRS)


all:		lib lib-opt


install:


clean:		subdirs
		@$(CleanObjFiles)


ifeq ($(objects),)

lib:		subdirs

lib-opt:	subdirs

lib-dbg:	subdirs

else

lib:		$(objects) subdirs
		@$(MakeLib)

lib-opt:	$(objects-opt) subdirs
		@$(MakeLibOpt)

lib-dbg:	$(objects-dbg) subdirs
		@$(MakeLibDbg)


endif

# Make the sub directories sequentially to avoid race conditions
# when the object files are gathered into the library.

subdirs:
		+@for i in $(SUBDIRS); do \
		  $(MAKE) -C $${i} $(MAKECMDGOALS); \
		done


include 	$(JEMPATH)/makefiles/obj.mk
