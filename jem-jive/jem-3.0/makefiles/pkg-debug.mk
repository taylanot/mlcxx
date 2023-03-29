
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


package		= $(shell "$${JEMDIR:?}"/bin/pkgname)
libdir		= $(JEMPATH)/lib
target		= $(libdir)/$(package)_debug$(OBJ_SUFFIX)

CMODE		= $(CMODE_DBG)


#########################################################################
# Rules
#########################################################################


.SUFFIXES:

.PHONY:		all lib clean


all:		$(target)

lib:		$(target)

clean:

$(target):	debug.cpp
		@test -d $(libdir) || $(MKDIR) $(libdir)
		$(Compile)
