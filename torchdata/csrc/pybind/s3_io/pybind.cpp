#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "s3_io.h"

namespace py = pybind11;
// TODO: change to S3Client
using torchdata::S3Init;
PYBIND11_MODULE(_torchdata, m) {
    py::class_<S3Init>(m, "S3Init")
        // TODO: pass in timeout
        .def(py::init<>())
        .def("s3_read",
             [](S3Init* self, const std::string& file_url) {
                 std::string result;
                 self->s3_read(file_url, &result);
                 return py::bytes(result);
             })
        .def("list_files",
             [](S3Init* self, const std::string& file_url) {
                 std::vector<std::string> filenames;
                 self->list_files(file_url, &filenames);
                 return filenames;
             })
        .def("file_exists",
             [](S3Init* self, const std::string& file_url) {
                 return self->file_exists(file_url);
             });
}
