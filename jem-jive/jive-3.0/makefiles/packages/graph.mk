
# Generated on Wed Dec 15 18:15:54 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JIVE_GRAPH_MAKEDEFS_MK
       JIVE_GRAPH_MAKEDEFS_MK = 1

ifndef JIVEPATH
  JIVEPATH := $(JIVEDIR)
endif

include $(JIVEPATH)/makefiles/jive.mk

include $(JIVEPATH)/makefiles/packages/util.mk

JIVE_PACKAGES += graph
JIVE_LIBS     := jivegraph $(JIVE_LIBS)

endif
