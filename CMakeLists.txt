cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Most of the configurations are taken from PyTorch
# https://github.com/pytorch/pytorch/blob/0c9fb4aff0d60eaadb04e4d5d099fb1e1d5701a9/CMakeLists.txt

# Use compiler ID "AppleClang" instead of "Clang" for XCode.
# Not setting this sometimes makes XCode C compiler gets detected as "Clang",
# even when the C++ one is detected as "AppleClang".
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)

# Suppress warning flags in default MSVC configuration.  It's not
# mandatory that we do this (and we don't if cmake is old), but it's
# nice when it's possible, and it's possible on our Windows configs.
if(NOT CMAKE_VERSION VERSION_LESS 3.15.0)
  cmake_policy(SET CMP0092 NEW)
endif()

project(_torchdata)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

# Options
option(BUILD_S3 "Build s3 io functionality" OFF)

find_package(Torch REQUIRED)
message(STATUS "BUILD_S3: ${BUILD_S3}")

if(BUILD_S3)
  message(STATUS "BUILD_S3: ${BUILD_S3}")
  find_package(Python3 COMPONENTS Interpreter Development)
  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  find_package(pybind11 REQUIRED)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(INCLUDE_DIRS "torchdata/csrc/pybind/s3_io")

  set(SOURCES "${INCLUDE_DIRS}/s3_io.cpp" )

  include_directories(${INCLUDE_DIRS})
  pybind11_add_module(_torchdata ${SOURCES} "${INCLUDE_DIRS}/pybind.cpp")

  message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")

  target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})
endif()
