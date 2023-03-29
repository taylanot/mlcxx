
# Generated on Wed Dec 15 18:15:54 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.

ifndef JIVE_GL_MAKEDEFS_MK
       JIVE_GL_MAKEDEFS_MK = 1

ifndef JIVEPATH
  JIVEPATH := $(JIVEDIR)
endif

include $(JIVEPATH)/makefiles/jive.mk

include $(JIVEPATH)/makefiles/packages/app.mk
include $(JIVEPATH)/makefiles/packages/fem.mk
include $(JIVEPATH)/makefiles/packages/geom.mk
include $(JIVEPATH)/makefiles/packages/util.mk
include $(JEMPATH)/makefiles/packages/gl.mk

JIVE_PACKAGES += gl
JIVE_LIBS     := jivegl $(JIVE_LIBS)

endif
