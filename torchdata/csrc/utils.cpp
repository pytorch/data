#include <torch/script.h>

namespace torchdata {

namespace {

bool is_s3_io_available() {
#ifdef INCLUDE_S3_IO
  return true;
#else
  return false;
#endif
}

} // namespace

TORCH_LIBRARY_FRAGMENT(torchdata, m) {
  m.def("torchdata::is_s3_io_available", &is_s3_io_available);
}

} // namespace torchdata
