#include <torch/extension.h>

#ifdef INCLUDE_S3_IO
// #include "S3Handler.h"
#include <torchdata/csrc/pybind/S3Handler/S3Handler.h>
using torchdata::S3Handler;
#endif

// #ifdef INCLUDE_S3_IO
// namespace py = pybind11;
// #endif
PYBIND11_MODULE(_torchdata, m)
{
#ifdef INCLUDE_S3_IO
    py::class_<S3Handler>(m, "S3Handler")
        .def(py::init<const long, const std::string &>())
        .def("s3_read",
             [](S3Handler *self, const std::string &file_url)
             {
                 std::string result;
                 self->S3Read(file_url, &result);
                 return py::bytes(result);
             })
        .def("list_files",
             [](S3Handler *self, const std::string &file_url)
             {
                 std::vector<std::string> filenames;
                 self->ListFiles(file_url, &filenames);
                 return filenames;
             })
        .def("set_buffer_size",
             [](S3Handler *self, const uint64_t buffer_size)
             {
                 self->SetBufferSize(buffer_size);
             })
        .def("set_multi_part_download",
             [](S3Handler *self, const bool multi_part_download)
             {
                 self->SetMultiPartDownload(multi_part_download);
             })
        .def("clear_marker",
             [](S3Handler *self)
             {
                 self->ClearMarker();
             })
        .def(py::pickle(
            [](const S3Handler &s3_handler) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(s3_handler.GetRequestTimeoutMs(),
                                      s3_handler.GetRegion(),
                                      s3_handler.GetLastMarker(),
                                      s3_handler.GetUseMultiPartDownload(),
                                      s3_handler.GetBufferSize());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 5)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                S3Handler s3_handler(t[0].cast<long>(), t[1].cast<std::string>());

                /* Assign any additional state */
                s3_handler.SetLastMarker(t[2].cast<std::string>());
                s3_handler.SetMultiPartDownload(t[3].cast<bool>());
                s3_handler.SetBufferSize(t[4].cast<int>());

                return s3_handler;
            }));
#endif
}
