#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "s3_io.h"

namespace py = pybind11;
using torchdata::S3Handler;
PYBIND11_MODULE(_torchdata, m)
{
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
        .def("set_max_keys",
             [](S3Handler *self, const int max_keys)
             {
                 self->SetMaxKeys(max_keys);
             })
        .def("clear_marker",
             [](S3Handler *self)
             {
                 self->ClearMarker();
             });
}
