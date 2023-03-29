
# Generated on Wed Dec 15 18:05:34 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.


ifndef JEM_MP_MAKEDEFS_MK
       JEM_MP_MAKEDEFS_MK = 1

ifndef JEMPATH
  JEMPATH := $(JEMDIR)
endif
include $(JEMPATH)/makefiles/packages/base.mk
include $(JEMPATH)/makefiles/packages/io.mk
include $(JEMPATH)/makefiles/packages/util.mk

JEM_PACKAGES += mp
JEM_LIBS     := jemmp $(JEM_LIBS)

SYS_LIBS     := mpi $(SYS_LIBS)
SYS_LIBDIRS  := /home/jem-jive/openmpi/lib $(SYS_LIBDIRS)
SYS_INCDIRS  := /home/jem-jive/openmpi/include $(SYS_INCDIRS)

endif
