# # compiler
# CXX := g++

# # building directory
# BUILD_DIR := build

# # flags for the compiler
# CXXFLAGS := -std=c++17 -Wall -fopenmp -lstdc++fs -lopenblas

# # include our headers
# INCLFLAGS := -I .
# CXXFLAGS  += $(INCLFLAGS)

# # libraries needed for mlcxx library
# LIBS_NAME +=  armadillo boost_serialization boost_program_options #boost_iostreams
# LIBS := $(addprefix -l,$(LIBS_NAME))

# # source files to be compiled
# SOURCES := $(wildcard *.cpp)

# # Generate a list of executable targets by removing the '.cpp' extension
# TARGETS := $(SOURCES:.cpp=)
# CLEAN_LIST := $(TARGETS) 

# all: $(TARGETS)

# # Rule to build each executable from its corresponding source file
# $(TARGETS): % : %.cpp 
# 	@mkdir -p $(BUILD_DIR)
# 	$(CXX) $(CXXFLAGS)  $< -o $(addprefix $(BUILD_DIR)/,$@) $(LDFLAGS) $(LIBS)

# # Rule for clearning
# clean:
# 	@rm -rfv $(BUILD_DIR)
#
################################################################################

# compiler
CXX := g++

# building directory
BUILD_DIR := build

# flags for the compiler
CXXFLAGS := -std=c++20 -Wall -fopenmp -lstdc++fs #-lhighs -lcurl #-lopenblas

# include our headers
INCLFLAGS := -I .
INCLFLAGS += -I /usr/local/include/highs/
CXXFLAGS  += $(INCLFLAGS)


# libraries needed for mlcxx library
# LIBS_NAME +=  armadillo boost_serialization boost_program_options highs curl #lapack #boost_iostreams
LIBS_NAME +=  armadillo highs curl # boost_program_options #lapack 
LIBS := $(addprefix -l,$(LIBS_NAME))

# source files to be compiled (search recursively for .cpp files)
SOURCES := $(shell find . -name '*.cpp')

# CXXFLAGS += -g -fsanitize=address

# Generate a list of executable targets by removing the '.cpp' extension
TARGETS := $(patsubst %.cpp, %, $(SOURCES))
CLEAN_LIST := $(TARGETS)

.PHONY: all clean

all: $(TARGETS)

$(TARGETS): % : %.cpp 
	@mkdir -p $(dir $(BUILD_DIR)/$@)
	$(CXX) $(CXXFLAGS) $< -o $(addprefix $(BUILD_DIR)/,$@) $(LDFLAGS) $(LIBS)

# Rule for cleaning
clean:
	@rm -rfv $(BUILD_DIR)


