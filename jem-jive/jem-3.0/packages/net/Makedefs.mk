
# Generated on Wed Dec 15 18:05:34 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.


ifndef JEM_NET_MAKEDEFS_MK
       JEM_NET_MAKEDEFS_MK = 1

ifndef JEMPATH
  JEMPATH := $(JEMDIR)
endif
include $(JEMPATH)/makefiles/packages/base.mk
include $(JEMPATH)/makefiles/packages/io.mk
include $(JEMPATH)/makefiles/packages/util.mk

JEM_PACKAGES += net
JEM_LIBS     := jemnet $(JEM_LIBS)


endif
