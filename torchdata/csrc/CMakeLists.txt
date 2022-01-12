if(BUILD_S3)
  message(STATUS "Building S3 IO functionality")
  find_package(Python3 COMPONENTS Interpreter Development)
  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  find_package(pybind11 REQUIRED)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(SOURCES "pybind/s3_io/s3_io.cpp" )

  pybind11_add_module(_torchdata ${SOURCES} "pybind/s3_io/pybind.cpp")

  message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")

  target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})
endif()
