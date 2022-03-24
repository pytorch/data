if(BUILD_S3)
  message(STATUS "Building S3 IO functionality")

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  if(BUILD_PYTHON_VERSION)
    find_package(Python3 ${BUILD_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
  endif()

  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  find_package(pybind11 CONFIG REQUIRED)
  include_directories(${pybind11_INCLUDE_DIRS})
  message(STATUS " Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

  message(STATUS " pybind11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
  message(STATUS " pybind11_LIBRARIES = ${pybind11_LIBRARIES}")


  set(S3HANDLER_INCLUDE_DIRS "pybind/S3Handler")
  include_directories(${S3HANDLER_INCLUDE_DIRS})

  set(SOURCES "pybind/S3Handler/S3Handler.cpp" )

  pybind11_add_module(_torchdata ${SOURCES} "pybind/S3Handler/pybind.cpp")

  message(STATUS " pybind11_INCLUDE_DIRS = ${pybind11_INCLUDE_DIRS}")
  message(STATUS " pybind11_LIBRARIES = ${pybind11_LIBRARIES}")

  message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")
  target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})
endif()
