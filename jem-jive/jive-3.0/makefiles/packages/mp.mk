
# Generated on Wed Dec 15 18:15:54 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JIVE_MP_MAKEDEFS_MK
       JIVE_MP_MAKEDEFS_MK = 1

ifndef JIVEPATH
  JIVEPATH := $(JIVEDIR)
endif

include $(JIVEPATH)/makefiles/jive.mk

include $(JIVEPATH)/makefiles/packages/util.mk

JIVE_PACKAGES += mp
JIVE_LIBS     := jivemp $(JIVE_LIBS)

endif
