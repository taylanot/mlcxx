ifndef JEM_DEFAULTS_MK
       JEM_DEFAULTS_MK = 1


#########################################################################
# Variables
#########################################################################


RM		= rm -f --
RM_R		= rm -f -r --
MKDIR		= mkdir --

OBJ_SUFFIX	= .o
EXE_SUFFIX	=
EXE_SUFFIX_OPT	= -opt$(EXE_SUFFIX)
EXE_SUFFIX_DBG	= -dbg$(EXE_SUFFIX)

LIB_PREFIX	= lib
LIB_SUFFIX	= .a
DLL_PREFIX	= lib
DLL_SUFFIX	= .so

OPT_LEVEL	= $(JEM_OPT_LEVEL)

JEM_LIBS	= jem
SYS_LIBS	=
SYS_LIBDIRS	=
SYS_INCDIRS	=
DBG_TOOLS	=

TEMP_DIRS	= $(OBJDIR) $(OBJDIR_OPT) $(OBJDIR_DBG)
CORE_DUMPS	= $(wildcard core) $(wildcard core.[0-9]*)

CXX		= $(error no compiler has been configured;\
                          run \$JEMDIR/configure first)


endif
