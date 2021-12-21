cmake_minimum_required(VERSION 3.13)
project(_torchdata)

set(CMAKE_CXX_STANDARD 17)
SET(TORCH_MIN_VERSION "1.5.1")

find_package(Python3 COMPONENTS Interpreter Development)

find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(INCLUDE_DIRS "torchdata/csrc/pybind/s3_io")

set(SOURCES "${INCLUDE_DIRS}/s3_io.cpp" )

include_directories(${INCLUDE_DIRS})
find_package(pybind11 REQUIRED)
pybind11_add_module(_torchdata ${SOURCES} "${INCLUDE_DIRS}/pybind.cpp")

Message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")

target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})


# project(torchdata)

# find_package(Python3 COMPONENTS Interpreter Development)
# find_package(AWSSDK RUNTIME COMPONENTS s3 transfer)
# find_package(pybind11 REQUIRED)

# set(INCLUDE_DIRS "torchdata/csrc/s3_io")
# set(SOURCES "${INCLUDE_DIRS}/s3_io.cpp")

# include_directories(${INCLUDE_DIRS})
# # add_subdirectory(torchdata/csrc)

# pybind11_add_module(_torchdata ${SOURCES} "${INCLUDE_DIRS}/pybind.cpp")

# target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})

# # EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# # define (VERSION_INFO) here.
# # target_compile_definitions(_torchdata PRIVATE )