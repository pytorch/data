add_library(
  torcharrow_udfs OBJECT
  functions.h
  string_functions.h
)

target_link_libraries(torcharrow_udfs f4d_functions_string f4d_functions_common)