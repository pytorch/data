# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(BUILD_S3)
  message(STATUS "Building S3 IO functionality")

  # To make the right CPython is built with on GitHub Actions,
  # see https://github.com/actions/setup-python/issues/121#issuecomment-1014500503
  set(Python_FIND_FRAMEWORK "LAST")

  if(WIN32)
      find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
  endif()

  # AWSSDK Dependencies
  if(WIN32)
    set(AWSSDK_DEP_LIBRARIES "Userenv;version;ws2_32;Bcrypt;Wininet;winhttp;Crypt32;Secur32;NCrypt;Shlwapi")
  elseif(APPLE)
    set(AWSSDK_DEP_LIBRARIES pthread curl)
  elseif(UNIX)
    set(AWSSDK_DEP_LIBRARIES pthread)
    if(STATIC_DEPS)
      set(OPENSSL_USE_STATIC_LIBS TRUE)
    endif()
    include(FindCURL)
    list(APPEND AWSSDK_DEP_LIBRARIES ${CURL_LIBRARIES})
    include(FindOpenSSL)
    list(APPEND AWSSDK_DEP_LIBRARIES ${OPENSSL_SSL_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARIES})
  endif()
  message(STATUS "AWSSDK DEPENDENCIES AWSSDK_DEP_LIBRARIES: ${AWSSDK_DEP_LIBRARIES}")

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(
    EXTENSION_SOURCES
    pybind/pybind.cpp
    pybind/S3Handler/S3Handler.cpp
  )

  add_library(_torchdata MODULE ${EXTENSION_SOURCES})

  if(NOT USE_SYSTEM_AWS_SDK_CPP)
    add_dependencies(_torchdata aws_sdk)
  endif()

  target_include_directories(
    _torchdata
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${Python_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIRS}
    ${AWSSDK_INCLUDE_DIRS}
    )

  target_link_libraries(
    _torchdata
    PRIVATE
    ${Python_LIBRARIES}
    ${ADDITIONAL_ITEMS}
    ${AWSSDK_LIBRARIES}
    ${AWSSDK_DEP_LIBRARIES}
    )

  set_target_properties(_torchdata PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(_torchdata PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  if (APPLE)
    set_target_properties(_torchdata PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(APPLE)

endif()
