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

  set(aws_cpp_sdk_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/aws-sdk-cpp)
  set(aws_cpp_sdk_INSTALL "${CMAKE_BINARY_DIR}/aws_cpp_sdk_INSTALL")
  set(AWSSDK_INCLUDE_DIRS "${aws_cpp_sdk_INSTALL}/include")

  set(
    AWSSDK_LIBS
    aws-cpp-sdk-transfer aws-cpp-sdk-s3 aws-cpp-sdk-core aws-crt-cpp
    aws-c-mqtt aws-c-event-stream aws-c-s3 aws-c-auth aws-c-http aws-c-io aws-c-compression aws-c-cal aws-c-sdkutils aws-checksums aws-c-common
  )

  foreach(lib ${AWSSDK_LIBS})
    if(WIN32)
      list(APPEND AWSSDK_LIBRARIES ${aws_cpp_sdk_INSTALL}/lib/${lib}.lib)
    else()
      list(APPEND AWSSDK_LIBRARIES ${aws_cpp_sdk_INSTALL}/lib/lib${lib}.a)
    endif()
  endforeach()

  if(WIN32)
    list(APPEND AWSSDK_LIBRARIES "Userenv;version;ws2_32;Bcrypt;Wininet;winhttp;Crypt32;Secur32;NCrypt;Shlwapi")
  elseif(APPLE)
    list(APPEND AWSSDK_LIBRARIES "pthread;curl")
  else()
    list(APPEND AWSSDK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libs2n.a")
    list(APPEND AWSSDK_LIBRARIES "pthread;crypto;ssl;z;curl")
  endif()

  include(ExternalProject)
  ExternalProject_Add(
    aws-cpp-sdk
    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/aws-cpp-sdk
    SOURCE_DIR ${aws_cpp_sdk_SOURCE_DIR}
    INSTALL_DIR ${aws_cpp_sdk_INSTALL}
    CMAKE_ARGS "-DBUILD_SHARED_LIBS=OFF" "-DBUILD_ONLY=transfer;s3" "-DENABLE_TESTING=OFF" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=${aws_cpp_sdk_INSTALL}"
    BUILD_BYPRODUCTS ${AWSSDK_LIBRARIES}
    )
endif()

message(STATUS "aws-sdk-cpp include dirs: " "${AWSSDK_INCLUDE_DIRS}")
