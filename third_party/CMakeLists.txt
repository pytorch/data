# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---[ pybind11
if(USE_SYSTEM_PYBIND11)
  find_package(pybind11 CONFIG)
  if(NOT pybind11_FOUND)
    find_package(pybind11)
  endif()
  if(NOT pybind11_FOUND)
    message(FATAL "Cannot find system pybind11")
  endif()
else()
  message(STATUS "Using third_party/pybind11.")
  set(pybind11_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/pybind11/include)
  install(DIRECTORY ${pybind11_INCLUDE_DIRS}
          DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.h")
endif()
message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
add_library(pybind::pybind11 INTERFACE IMPORTED)
set_property(TARGET pybind::pybind11 PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${pybind11_INCLUDE_DIRS})
set_property(TARGET pybind::pybind11 PROPERTY
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${pybind11_INCLUDE_DIRS})
set_property(TARGET pybind::pybind11 PROPERTY
    INTERFACE_LINK_LIBRARIES python::python)

# ---[ aws-sdk-cpp
if(USE_SYSTEM_AWS_SDK_CPP)
  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  if(NOT AWSSDK_FOUND)
    message(FATAL "Cannot find system aws-sdk-cpp")
  endif()
else()
  message(STATUS "Using third_party/aws-sdk-cpp.")

  set(AWS_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/aws-sdk-cpp)
  set(AWS_INSTALL "${CMAKE_BINARY_DIR}/aws_install")
  set(AWS_INCLUDE_PATH "${AWS_INSTALL}/include")

  list(APPEND AWS_LIBS "${AWS_INSTALL}/lib/libaws-cpp-sdk-s3.so")
  list(APPEND AWS_LIBS "${AWS_INSTALL}/lib/libaws-cpp-sdk-transfer.so")

  include(ExternalProject)
  ExternalProject_Add(
    aws-cpp-sdk
    PREFIX ${CMAKE_BINARY_DIR}/aws-cpp-sdk
    SOURCE_DIR ${AWS_SOURCE_DIR}
    URL ""
    INSTALL_DIR ${AWS_INSTALL}
    CMAKE_ARGS "-DBUILD_ONLY=transfer;s3" "-DENABLE_TESTING=OFF" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=${AWS_INSTALL}"
    BUILD_BYPRODUCTS ${AWS_LIBS}
    )

  message(STATUS "aws-sdk-cpp include dirs: " "${AWS_INCLUDE_PATH}")
endif()
