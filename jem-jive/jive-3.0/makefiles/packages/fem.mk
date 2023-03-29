
# Generated on Wed Dec 15 18:15:54 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JIVE_FEM_MAKEDEFS_MK
       JIVE_FEM_MAKEDEFS_MK = 1

ifndef JIVEPATH
  JIVEPATH := $(JIVEDIR)
endif

include $(JIVEPATH)/makefiles/jive.mk

include $(JIVEPATH)/makefiles/packages/algebra.mk
include $(JIVEPATH)/makefiles/packages/app.mk
include $(JIVEPATH)/makefiles/packages/model.mk
include $(JIVEPATH)/makefiles/packages/geom.mk
include $(JIVEPATH)/makefiles/packages/graph.mk
include $(JIVEPATH)/makefiles/packages/mp.mk
include $(JIVEPATH)/makefiles/packages/util.mk

JIVE_PACKAGES += fem
JIVE_LIBS     := jivefem $(JIVE_LIBS)

endif
