program = mlcxx

#subdirs = utils src model algo tests stats opt

#include $(JEMDIR)/makefiles/prog.mk
#include $(JEMDIR)/makefiles/packages/*.mk
include $(JIVEDIR)/makefiles/prog.mk
include $(JIVEDIR)/makefiles/packages/*.mk


LIBS +=  armadillo boost_serialization boost_program_options #torch_cpu torch c10 
#MY_LIBDIRS += $(TORCHDIR)/lib
#INCDIRS += ${TORCHDIR}/include ${TORCHDIR}/include/torch/csrc/api/include
CXX_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0 -std=gnu++17 -Wl,-R${TORCHDIR}/lib -fopenmp -lstdc++fs
MY_INCDIRS = /home/shared/tools/mlcxx-updated

