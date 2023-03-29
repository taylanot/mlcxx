
# Generated on Wed Dec 15 18:05:34 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.


ifndef JEM_XUTIL_MAKEDEFS_MK
       JEM_XUTIL_MAKEDEFS_MK = 1

ifndef JEMPATH
  JEMPATH := $(JEMDIR)
endif
include $(JEMPATH)/makefiles/packages/util.mk

JEM_PACKAGES += xutil
JEM_LIBS     := jemxutil $(JEM_LIBS)


endif
