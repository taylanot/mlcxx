program = mlcxx

#subdirs = utils src model algo data  

#include $(JEMDIR)/makefiles/prog.mk
#include $(JEMDIR)/makefiles/packages/*.mk
include $(JIVEDIR)/makefiles/prog.mk
include $(JIVEDIR)/makefiles/packages/*.mk


#include $(JIVEDIR)/makefiles/packages/app.mk
#include $(JIVEDIR)/makefiles/packages/fem.mk
#include $(JIVEDIR)/makefiles/prog.mk

#include $(TORCHDIR)/include
#include $(TORCHDIR)/include/torch/csrc/api/include/
#MY_LIBS += boost_filesystem boost_system
#LIBS += torch_cpu torch armadillo mlpack boost_serialization boost_filesystem boost_system boost_program_options c10
LIBS += torch_cpu torch armadillo mlpack boost_serialization boost_program_options c10 
MY_LIBDIRS += $(TORCHDIR)/lib
INCDIRS += ${TORCHDIR}/include ${TORCHDIR}/include/torch/csrc/api/include
CXX_FLAGS += -D_GLIBCXX_USE_CXX11_ABI=0 -std=gnu++17 -Wl,-R${TORCHDIR}/lib -fopenmp -lstdc++fs
MY_INCDIRS = /home/shared/tools/mlcxx

#g++ --verbose example.cpp -I${TORCHDIR}/include -I${TORCHDIR}/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=0 -std=gnu++14 -L${TORCHDIR}/lib -Wl,-R${TORCHDIR}/lib -ltorch -ltorch_cpu -lc10 -o example
#
#all: example
#LDP=-L ${TORCHDIR}/lib
#LIBS=torch_cpu torch c10
#LIB=$(patsubst %,-l%,$(LIBS))
#FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 -std=gnu++14 -Wl,-R${TORCHDIR}/lib
#INCS=${TORCHDIR}/include ${TORCHDIR}/include/torch/csrc/api/include
#INCLUDE=$(patsubst %,-I%,$(INCS))
#
#%: %.cpp
#	$(CXX) $< $(INCLUDE) $(FLAGS) $(LDP) $(LIB)  -o $@ 
