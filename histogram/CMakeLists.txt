cmake_minimum_required(VERSION 3.2)
PROJECT(histogram)

# CUDA package
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES("../common/inc/")

# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)
source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})

CUDA_ADD_EXECUTABLE(histogram ${CURRENT_HEADERS} ${CURRENT_SOURCES})