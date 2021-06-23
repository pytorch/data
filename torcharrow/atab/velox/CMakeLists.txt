# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.5)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(TorchArrow-Velox)

set(F4D_ROOT ${CMAKE_BINARY_DIR}/f4d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mavx -mf16c -masm=intel -mlzcnt")

# Setup CCache
# Reference: https://github.com/pytorch/pytorch/blob/c03cae49fc70408a93f4de15a11a5a73997e9565/CMakeLists.txt#L302-L311
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
  message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
endif()

# Find Python interpreter, compiler and development environment (include directories and libraries).
# Copied from sadboy's CMakeLists.txt: https://github.com/sadboy/f4d/blob/57bbc67c3501d0a6a88d71d81bac271fadea9e88/torcharrow/CMakeLists.txt#L14
find_package(Python COMPONENTS Interpreter Development)

include_directories(f4d)
# Otherwise some targets, such as dwrf-proto-wrapper.cpp.o
# will complain cannot find files to be included
# Reference: https://github.com/facebookexternal/presto_cpp/blob/f278b9b186d9753dc59f58f31bd09db1c89c3c2b/CMakeLists.txt#L104
include_directories(${F4D_ROOT})

add_subdirectory(f4d)
add_subdirectory(pybind11)
add_subdirectory(functions)

# Define our Python module:
pybind11_add_module(
  _torcharrow
  MODULE
  lib.cpp
  vector.h
  vector.cpp
  column.h
  column.cpp)

# TODO: Need this to match with f4d, however pybind11 requires "hidden" visibility
# Copied from sadboy's CMakeLists.txt: https://github.com/sadboy/f4d/blob/57bbc67c3501d0a6a88d71d81bac271fadea9e88/torcharrow/CMakeLists.txt#L29
set_target_properties(_torcharrow PROPERTIES CXX_VISIBILITY_PRESET default)

# Link with Velox:
target_link_libraries(_torcharrow PRIVATE
  f4d_vector
  f4d_memory
  f4d_core
  f4d_functions_common
  f4d_parse_expression
  f4d_parse_parser)

target_compile_definitions(_torcharrow PRIVATE VERSION_INFO=0.0.1)
