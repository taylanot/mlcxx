
# Generated on Wed Dec 15 18:05:33 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.


ifndef JEM_IO_MAKEDEFS_MK
       JEM_IO_MAKEDEFS_MK = 1

ifndef JEMPATH
  JEMPATH := $(JEMDIR)
endif
include $(JEMPATH)/makefiles/packages/base.mk
include $(JEMPATH)/makefiles/packages/util.mk

JEM_PACKAGES += io

SYS_LIBS     := z  $(SYS_LIBS)
SYS_LIBDIRS  :=   $(SYS_LIBDIRS)
SYS_INCDIRS  :=   $(SYS_INCDIRS)

endif
