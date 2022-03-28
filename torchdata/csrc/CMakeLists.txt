if(BUILD_S3)

  message(STATUS "Building S3 IO functionality")

  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  find_package(pybind11 REQUIRED)

  if(BUILD_PYTHON_VERSION)
    find_package(Python3 ${BUILD_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
  else()
    find_package(Python3 COMPONENTS Interpreter Development)
  endif()

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  add_library(_torchdata SHARED "pybind/S3Handler/S3Handler.cpp")
	# target_include_directories(_torchdata PRIVATE ${PROJECT_SOURCE_DIR} ${Python_INCLUDE_DIR})
	target_include_directories(_torchdata PRIVATE ${PROJECT_SOURCE_DIR})

  message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")
  target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS} Python3::Python)

  set_target_properties(_torchdata PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(_torchdata PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  if (APPLE)
    set_target_properties(_torchdata PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(APPLE)

  install(
    TARGETS _torchdata
    LIBRARY DESTINATION .
    RUNTIME DESTINATION . # For Windows
    )

endif()
