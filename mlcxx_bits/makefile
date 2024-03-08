# compiler
CXX := g++

# building directory
BUILD_DIR := build

# flags for the compiler
CXXFLAGS := -std=c++17 -Wall -fopenmp -lstdc++fs -lopenblas

# include our headers
INCLFLAGS := -I .
CXXFLAGS  += $(INCLFLAGS)

# libraries needed for mlcxx library
LIBS_NAME +=  armadillo boost_serialization boost_program_options 
LIBS := $(addprefix -l,$(LIBS_NAME))

# source files to be compiled
SOURCES := $(wildcard *.cpp)

# Generate a list of executable targets by removing the '.cpp' extension
TARGETS := $(SOURCES:.cpp=)
CLEAN_LIST := $(TARGETS) 

all: $(TARGETS)

# Rule to build each executable from its corresponding source file
$(TARGETS): % : %.cpp 
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS)  $< -o $(addprefix $(BUILD_DIR)/,$@) $(LDFLAGS) $(LIBS)

# Rule for clearning
clean:
	@rm -rfv $(BUILD_DIR)

