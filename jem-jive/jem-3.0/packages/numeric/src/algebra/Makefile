
JEMPATH := ../../../..

include $(JEMPATH)/packages/numeric/Makedefs.mk
include $(JEMPATH)/makefiles/private.mk
include $(JEMPATH)/makefiles/pkg-src.mk


ifeq ($(CXX_NAME),icc)

$(OBJBASE_OPT)utilities$(OBJ_SUFFIX): 	OPT_LEVEL = 3
$(OBJBASE_OPT)LUSolver$(OBJ_SUFFIX): 	OPT_LEVEL = 3
$(OBJBASE_OPT)Cholesky$(OBJ_SUFFIX): 	OPT_LEVEL = 3

endif
