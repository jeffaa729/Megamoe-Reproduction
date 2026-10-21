cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90a)
endif()
project(megamoe LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(CUDAToolkit REQUIRED)
enable_testing()

set(GPU_KERNELS_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Benchmark Python environment")
set(CUTLASS_DIR "$ENV{CUTLASS_DIR}" CACHE PATH "CUTLASS checkout or installed DeepGEMM package")
if(NOT CUTLASS_DIR)
    execute_process(COMMAND "${GPU_KERNELS_PYTHON}" -c
        "import importlib.util, pathlib; s = importlib.util.find_spec('deep_gemm'); print(pathlib.Path(s.origin).parent if s else '')"
        OUTPUT_VARIABLE CUTLASS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_path(MEGAMOE_CUTLASS_INCLUDE_DIR cute/tensor.hpp
    HINTS "${CUTLASS_DIR}/include" "${CUTLASS_DIR}" NO_DEFAULT_PATH REQUIRED)

add_library(megamoe_common STATIC kernels/common/cuda_common.cu)
target_include_directories(megamoe_common PUBLIC include)
target_link_libraries(megamoe_common PUBLIC CUDA::cudart CUDA::cuda_driver)
set_target_properties(megamoe_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Preserve the library names consumed by the ctypes benchmark wrappers.
add_library(gpu_kernels_moe_swiglu_bench SHARED
    kernels/moe/swiglu.cu benchmarks/bindings/moe_swiglu_bridge.cu)
target_link_libraries(gpu_kernels_moe_swiglu_bench PRIVATE megamoe_common)

add_library(gpu_kernels_fused_moe_bench SHARED
    kernels/moe/fused/sm90.cu benchmarks/bindings/fused_moe_bridge.cu)
target_include_directories(gpu_kernels_fused_moe_bench PRIVATE "${MEGAMOE_CUTLASS_INCLUDE_DIR}")
target_link_libraries(gpu_kernels_fused_moe_bench PRIVATE megamoe_common)
target_compile_options(gpu_kernels_fused_moe_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--expt-relaxed-constexpr;-Xptxas=-v>)
set_target_properties(gpu_kernels_fused_moe_bench PROPERTIES CUDA_ARCHITECTURES 90a)

if(EXISTS "${GPU_KERNELS_PYTHON}")
    add_test(NAME moe_reference COMMAND "${GPU_KERNELS_PYTHON}"
        -m unittest discover -s tests -p test_moe_reference.py)
    set_tests_properties(moe_reference PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
