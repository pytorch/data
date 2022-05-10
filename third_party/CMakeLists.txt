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

  if(WIN32)
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-cpp-sdk-transfer.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-cpp-sdk-s3.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-cpp-sdk-core.lib")

    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-crt-cpp.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-mqtt.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-event-stream.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-s3.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-auth.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-http.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-io.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-compression.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-cal.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-sdkutils.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-checksums.lib")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/aws-c-common.lib")

    list(APPEND AWSSDK_LINK_LIBRARIES "Userenv;version;ws2_32;Bcrypt;Wininet;winhttp;Crypt32;Secur32;NCrypt;Shlwapi")
  elseif(APPLE)
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-cpp-sdk-transfer.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-cpp-sdk-s3.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-cpp-sdk-core.a")

    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-crt-cpp.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-mqtt.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-event-stream.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-s3.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-auth.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-http.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-io.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-compression.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-cal.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-sdkutils.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-checksums.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-common.a")

    list(APPEND AWSSDK_LINK_LIBRARIES "pthread;curl")
  else()
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-cpp-sdk-transfer.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-cpp-sdk-s3.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-cpp-sdk-core.a")

    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-crt-cpp.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-mqtt.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-event-stream.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-s3.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-auth.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-http.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-io.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libs2n.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-compression.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-cal.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-sdkutils.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-checksums.a")
    list(APPEND AWSSDK_LINK_LIBRARIES "${aws_cpp_sdk_INSTALL}/lib/libaws-c-common.a")

    list(APPEND AWSSDK_LINK_LIBRARIES "pthread;crypto;ssl;z;curl")
  endif()

  include(ExternalProject)
  ExternalProject_Add(
    aws-cpp-sdk
    PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/aws-cpp-sdk
    SOURCE_DIR ${aws_cpp_sdk_SOURCE_DIR}
    INSTALL_DIR ${aws_cpp_sdk_INSTALL}
    CMAKE_ARGS "-DBUILD_SHARED_LIBS=OFF" "-DBUILD_ONLY=transfer;s3" "-DENABLE_TESTING=OFF" "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_INSTALL_PREFIX=${aws_cpp_sdk_INSTALL}"
    BUILD_BYPRODUCTS ${AWSSDK_LINK_LIBRARIES}
    )
endif()

message(STATUS "aws-sdk-cpp include dirs: " "${AWSSDK_INCLUDE_DIRS}")
