cmake_minimum_required(VERSION 3.2)
PROJECT(performance)

FIND_PACKAGE(CUDA)

set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS *.h *.hpp *cuh)
file(GLOB CURRENT_SOURCES *.c *.cpp *.cu)
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

CUDA_ADD_EXECUTABLE(performance ${CURRENT_HEADERS} ${CURRENT_SOURCES})
