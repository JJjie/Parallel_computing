CUDA_PATH ?=/usr/local/cuda-8.0
NVCC      :=$(CUDA_PATH)/bin/nvcc -ccbin g++
INCLUDE   :=-I/usr/local/cuda-8.0/include/\
           -I/usr/local/cuda/samples/common/inc\
        -I/usr/include/c++\
        -I./\
        -I../common/inc

LIBRARIES :=-L/usr/local/cuda/lib64 -lcudart -lcufft
TARGETS   :=histogram
OBJECTS   :=histogram.o

.SUFFIXES:.o .cu .cpp
.cu.o:
	$(NVCC) -arch=sm_20  $(INCLUDE) -c -g -o $@  $< $(LIBRARIES)
.cpp.o:
	$(CXX) $(INCLUDE) -c -g -o $@ $< $(LIBRARIES)

all: $(TARGETS)

$(TARGETS): $(OBJECTS)
    #sudo cp /usr/local/cuda/lib64/libcufft.so.7.0 /usr/lib
	g++    $(INCLUDE) -g -o $@ $^ $(LIBRARIES)
run:
	./$(TARGETS)

clean:
	rm -rf *.o histogram