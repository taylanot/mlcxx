# compiler
CXX := g++

# building directory
BUILD_DIR := build

# flags for the compiler
CXXFLAGS := -std=c++23 -Wall -fopenmp -lstdc++fs 

# include our headers
INCLFLAGS := -I .
CXXFLAGS  += $(INCLFLAGS)

# libraries needed for mlcxx library
LIBS_NAME +=  armadillo curl 
LIBS := $(addprefix -l,$(LIBS_NAME))

# source files to be compiled (search recursively for .cpp files)
SOURCES := $(shell find . -name '*.cpp')

# Generate a list of executable targets by removing the '.cpp' extension
TARGETS := $(patsubst %.cpp, %, $(SOURCES))
CLEAN_LIST := $(TARGETS)

.PHONY: all clean

all: $(TARGETS)

# default rule for building targets
$(TARGETS): % : %.cpp 
	@mkdir -p $(dir $(BUILD_DIR)/$@)
	$(CXX) $(CXXFLAGS) $(if $(filter test.cpp,$<),-DMLPACK_ENABLE_ANN_SERIALIZATION,) $< -o $(addprefix $(BUILD_DIR)/,$@) $(LDFLAGS) $(LIBS)

# Rule for cleaning
clean:
	@rm -rfv $(BUILD_DIR)
