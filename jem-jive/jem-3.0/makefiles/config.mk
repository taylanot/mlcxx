
# Generated on Wed Dec 15 18:05:32 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JEM_CONFIG_MK
       JEM_CONFIG_MK = 1

SHELL               = /bin/dash

OBJ_SUFFIX          = .o
EXE_SUFFIX          = 
LIB_PREFIX          = lib
LIB_SUFFIX          = .a
DLL_PREFIX          = lib
DLL_SUFFIX          = .so

SYS_NAME            = linux
SYS_ARCH            = x86_64
SYS_BITS            = 64
SYS_LIBS            =  pthread m
SYS_LIBDIRS         = 
SYS_INCDIRS         = 

CXX                 = g++
CXX_NAME            = gcc
CXX_VERSION         = 90300

JEM_CXX_STD_FLAGS   = -O -Wall -fstrict-aliasing -m64 -msse2 -mtune=native
JEM_CXX_BLD_FLAGS   = -Wconversion -finline-functions -Wsuggest-override
JEM_CXX_LNK_FLAGS   = -rdynamic
JEM_CXX_DLL_FLAGS   = -fPIC
JEM_CXX_LIB_FLAGS   = 
JEM_CXX_DBG_FLAGS   = -g
JEM_CXX_PRF_FLAGS   = -pg
JEM_CXX_OPT_FLAGS1  = -O1 -DNDEBUG -funroll-loops
JEM_CXX_OPT_FLAGS2  = -O2 -DNDEBUG -funroll-loops
JEM_CXX_OPT_FLAGS3  = -O3 -DNDEBUG -funroll-loops -ftree-vectorize

JEM_OPT_LEVEL       = 3

endif
