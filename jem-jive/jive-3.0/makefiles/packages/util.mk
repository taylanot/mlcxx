
# Generated on Wed Dec 15 18:15:54 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JIVE_UTIL_MAKEDEFS_MK
       JIVE_UTIL_MAKEDEFS_MK = 1

ifndef JIVEPATH
  JIVEPATH := $(JIVEDIR)
endif

include $(JIVEPATH)/makefiles/jive.mk


JIVE_PACKAGES += util
JIVE_LIBS     := jiveutil $(JIVE_LIBS)

endif
