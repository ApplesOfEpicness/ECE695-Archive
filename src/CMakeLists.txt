# enable seperate compillation for dynamic parallelism
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include)

# CUDA Libaraies
cuda_add_library(cudaLib cudaLib.cu)
target_link_libraries(cudaLib ${CUDA_LIBRARIES})

# GCC (CPU) Libraries
add_library(cpuLib cpuLib.cpp )
