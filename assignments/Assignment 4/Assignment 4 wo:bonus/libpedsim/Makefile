TARGET = libpedsim.so
SOURCES = $(shell echo *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
CUDA_SOURCES = $(shell echo *.cu)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.co)
CXXFLAGS = -fPIC -shared -lm -fopenmp -march=native
CUDA_NVCC_FLAGS = --compiler-options -fPIC,-shared -Xcompiler -fopenmp -Xcompiler -march=native

all: $(TARGET)

debug: CXXFLAGS += -g
debug: CUDA_NVCC_FLAGS += -g -G
debug: all

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(CXX) $(FLAGS) $(CXXFLAGS) $(DEBUGFLAGS) -o $(TARGET) $(OBJECTS) $(CUDA_OBJECTS)

%.co: %.cu
	nvcc $(CUDA_NVCC_FLAGS) $(DEBUGFLAGS) -c -o $@ $<


clean:
	-rm $(TARGET) $(OBJECTS) $(CUDA_OBJECTS)
