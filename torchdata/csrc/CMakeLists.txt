if(BUILD_S3)
  message(STATUS "Building S3 IO functionality")
  list(APPEND CMAKE_INSTALL_PREFIX ";C:\\Program Files (x86)\\aws-cpp-sdk-all")
  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  find_package(pybind11 REQUIRED)
  if(BUILD_PYTHON_VERSION)
    find_package(Python3 ${BUILD_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(SOURCES "pybind/S3Handler/S3Handler.cpp" )

  pybind11_add_module(_torchdata ${SOURCES} "pybind/S3Handler/pybind.cpp")

  message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")

  target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})
endif()
