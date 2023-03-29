
# Generated on Wed Dec 15 18:05:34 CET 2021.
# CHANGES TO THIS FILE WILL BE LOST.


ifndef JEM_SSL_MAKEDEFS_MK
       JEM_SSL_MAKEDEFS_MK = 1

ifndef JEMPATH
  JEMPATH := $(JEMDIR)
endif
include $(JEMPATH)/makefiles/packages/base.mk
include $(JEMPATH)/makefiles/packages/io.mk
include $(JEMPATH)/makefiles/packages/util.mk
include $(JEMPATH)/makefiles/packages/net.mk
include $(JEMPATH)/makefiles/packages/crypto.mk

JEM_PACKAGES += ssl
JEM_LIBS     := jemssl $(JEM_LIBS)

SYS_LIBS     := ssl $(SYS_LIBS)

endif
