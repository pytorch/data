#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "s3_io.h"

namespace py = pybind11;
// TODO: change to S3Client
using torchdata::S3Handler;
PYBIND11_MODULE(_torchdata, m)
{
    py::class_<S3Handler>(m, "S3Handler")
        // TODO: pass in timeout
        .def(py::init<>())
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
             });
}
