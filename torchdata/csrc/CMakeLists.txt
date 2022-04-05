if(BUILD_S3)

  message(STATUS "Building S3 IO functionality")

  # To make the right CPython is built with on GitHub Actions,
  # see https://github.com/actions/setup-python/issues/121#issuecomment-1014500503
  set(Python_FIND_FRAMEWORK "LAST")

  if (WIN32)
      find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
  endif()

  list(APPEND CMAKE_PREFIX_PATH ${AWSSDK_DIR})
  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  find_package(pybind11 CONFIG REQUIRED)

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(
    EXTENSION_SOURCES
    pybind/pybind.cpp
    pybind/S3Handler/S3Handler.cpp
  )

  add_library(_torchdata MODULE ${EXTENSION_SOURCES})

  target_include_directories(_torchdata PRIVATE ${PROJECT_SOURCE_DIR} ${Python_INCLUDE_DIR} {pybind11_INCLUDE_DIRS})

  message(STATUS "AWSSDK linked libs: ${AWSSDK_LINK_LIBRARIES}")
  target_link_libraries(
    _torchdata
    PRIVATE
    pybind11::module
    ${AWSSDK_LINK_LIBRARIES}
    ${Python_LIBRARIES}
    ${ADDITIONAL_ITEMS}
    )

  set_target_properties(_torchdata PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(_torchdata PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  if (APPLE)
    set_target_properties(_torchdata PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(APPLE)

endif()
