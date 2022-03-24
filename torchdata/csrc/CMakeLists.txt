################################################################################
# libtorchdata
################################################################################
set(
  LIBTORCHDATA_SOURCES
  )

set(
  LIBTORCHDATA_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}
  )

set(
  LIBTORCHDATA_LINK_LIBRARIES
  torch
  )

set(
  LIBTORCHDATA_COMPILE_DEFINITIONS)

#------------------------------------------------------------------------------#
# START OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#
if(BUILD_S3)
  find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  list(
    APPEND
    LIBTORCHDATA_LINK_LIBRARIES
    ${AWSSDK_LINK_LIBRARIES}
    ${AWSSDK_PLATFORM_DEPS}
    )
endif()
#------------------------------------------------------------------------------#
# END OF CUSTOMIZATION LOGICS
#------------------------------------------------------------------------------#
function (define_library name source include_dirs link_libraries compile_defs)
  add_library(${name} SHARED ${source})
  target_include_directories(${name} PRIVATE ${include_dirs})
  target_link_libraries(${name} ${link_libraries})
  target_compile_definitions(${name} PRIVATE ${compile_defs})
  set_target_properties(${name} PROPERTIES PREFIX "")
  if (MSVC)
    set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  endif(MSVC)
  install(
    TARGETS ${name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib  # For Windows
    )
endfunction()

define_library(
  libtorchdata
  "${LIBTORCHDATA_SOURCES}"
  "${LIBTORCHDATA_INCLUDE_DIRS}"
  "${LIBTORCHDATA_LINK_LIBRARIES}"
  "${LIBTORCHDATA_COMPILE_DEFINITIONS}"
  )


################################################################################
# _torchdata.so
################################################################################
if (BUILD_TORCHDATA_PYTHON_EXTENSION)
  # See https://github.com/pytorch/pytorch/issues/38122
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
  if (WIN32)
    find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)
    set(ADDITIONAL_ITEMS Python3::Python)
  endif()
  function(define_extension name sources libraries definitions)
    add_library(${name} SHARED ${sources})
    target_compile_definitions(${name} PRIVATE "${definitions}")
    target_include_directories(
      ${name} PRIVATE ${PROJECT_SOURCE_DIR} ${Python_INCLUDE_DIR})
    target_link_libraries(
      ${name}
      ${libraries}
      ${TORCH_PYTHON_LIBRARY}
      ${ADDITIONAL_ITEMS}
      )
    set_target_properties(${name} PROPERTIES PREFIX "")
    if (MSVC)
      set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
    endif(MSVC)
    if (APPLE)
      # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
      # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
      set_target_properties(${name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    install(
      TARGETS ${name}
      LIBRARY DESTINATION .
      RUNTIME DESTINATION .  # For Windows
      )
  endfunction()

  set(
    EXTENSION_SOURCES
    pybind/pybind.cpp
    )
  #----------------------------------------------------------------------------#
  # START OF CUSTOMIZATION LOGICS
  #----------------------------------------------------------------------------#
  if(BUILD_S3)
    message(STATUS "Setting up S3 IO building specifications")
    find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)

    list(
      APPEND
      EXTENSION_SOURCES
      pybind/S3Handler/S3Handler.cpp
      )
  endif()
  #----------------------------------------------------------------------------#
  # END OF CUSTOMIZATION LOGICS
  #----------------------------------------------------------------------------#
  define_extension(
    _torchdata
    "${EXTENSION_SOURCES}"
    libtorchdata
    "${LIBTORCHDATA_COMPILE_DEFINITIONS}"
    )
endif()

# if(BUILD_S3)
  # message(STATUS "Building S3 IO functionality")

  # find_package(AWSSDK REQUIRED COMPONENTS s3 transfer)
  # See https://github.com/pytorch/pytorch/issues/38122
  # find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

  # set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # if(BUILD_PYTHON_VERSION)
  #   find_package(Python3 ${BUILD_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
  # else()
  #   find_package(Python3 COMPONENTS Interpreter Development)
  # endif()
  # if (WIN32)
  #   set(ADDITIONAL_ITEMS Python3::Python)
  # endif()

  # function(define_extension name sources libraries definitions)
  #   add_library(${name} SHARED ${sources})
  #   target_compile_definitions(${name} PRIVATE "${definitions}")
  #   target_include_directories(
  #     ${name} PRIVATE ${PROJECT_SOURCE_DIR} ${Python_INCLUDE_DIR})
  #   target_link_libraries(
  #     ${name}
  #     ${libraries}
  #     ${TORCH_PYTHON_LIBRARY}
  #     ${ADDITIONAL_ITEMS}
  #     )
  #   set_target_properties(${name} PROPERTIES PREFIX "")
  #   if (MSVC)
  #     set_target_properties(${name} PROPERTIES SUFFIX ".pyd")
  #   endif(MSVC)
  #   if (APPLE)
  #     # https://github.com/facebookarchive/caffe2/issues/854#issuecomment-364538485
  #     # https://github.com/pytorch/pytorch/commit/73f6715f4725a0723d8171d3131e09ac7abf0666
  #     set_target_properties(${name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  #   endif()
  #   install(
  #     TARGETS ${name}
  #     LIBRARY DESTINATION .
  #     RUNTIME DESTINATION .  # For Windows
  #     )
  # endfunction()

  # define_extension(
  #   _torchdata
  #   "${EXTENSION_SOURCES}"
  #   libtorchdata
  #   "${LIBTORCHDATA_COMPILE_DEFINITIONS}"
  #   )

  # message(STATUS "All linked libs: ${AWSSDK_LINK_LIBRARIES}")
  # target_link_libraries(_torchdata PRIVATE ${AWSSDK_LINK_LIBRARIES} ${AWSSDK_PLATFORM_DEPS})
# endif()
