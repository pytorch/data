#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "S3Handler.h"

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
                                      s3_handler.GetMaxKeys(),
                                      s3_handler.GetUseMultiPartDownload(),
                                      s3_handler.GetBufferSize());
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                S3Handler s3_handler(t[0].cast<long>(), t[1].cast<std::string>());

                /* Assign any additional state */
                s3_handler.SetMaxKeys(t[4].cast<int>());

                return s3_handler;
            }));
    // .def(py::pickle(
    //     [](const Pickleable &p) { // __getstate__
    //         /* Return a tuple that fully encodes the state of the object */
    //         return py::make_tuple(p.value(), p.extra());
    //     },
    //     [](py::tuple t) { // __setstate__
    //         if (t.size() != 2)
    //             throw std::runtime_error("Invalid state!");

    //         /* Create a new C++ instance */
    //         Pickleable p(t[0].cast<std::string>());

    //         /* Assign any additional state */
    //         p.setExtra(t[1].cast<int>());

    //         return p;
    //     }));
}
