
ifndef JEM_MK
  $(error the makefile $$(JEMDIR)/makefiles/jem.mk\
          has not been included)
endif


#########################################################################
# Variables
#########################################################################


ifndef programs
  programs	:= $(wildcard *.cpp)
endif

programs	:= $(basename       $(programs))
programs-opt	:= $(addsuffix .opt,$(programs))
programs-dbg	:= $(addsuffix .dbg,$(programs))
programs	:= $(addsuffix .exe,$(programs))


RM_FILES	= $(CORE_DUMPS)\
		  $(wildcard *.exe)\
		  $(wildcard *.opt)\
		  $(wildcard *.dbg)


#########################################################################
# Rules
#########################################################################


.SUFFIXES:

.PHONY:		all opt debug install clean clean-all


all:		$(programs)
		+@$(MakeSubdirs) -t all

opt:		$(programs-opt)
		+@$(MakeSubdirs) -t opt

debug:		$(programs-dbg)
		+@$(MakeSubdirs) -t debug

install:


ifeq ($(strip $(RM_FILES)),)

clean:
		+@$(MakeSubdirs) -t clean

else

clean:
		@$(RM) $(RM_FILES)
		+@$(MakeSubdirs) -t clean

endif

clean-all:	clean


%.exe:		%.cpp
		$(CompileExe)

%.opt:		%.cpp
		$(CompileExe)

%.dbg:		%.cpp
		$(CompileExe)

%.opt:		CMODE = $(CMODE_OPT)

%.dbg:		CMODE = $(CMODE_DBG)

