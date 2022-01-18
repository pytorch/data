# Copyright (c) Facebook, Inc. and its affiliates.
import http.server
import os
import socketserver
import tempfile
import threading
import time
import unittest
import warnings

from functools import partial

import expecttest
from torch.testing._internal.common_utils import slowTest
from torchdata.datapipes.iter import FileLister, FileOpener, HttpReader, IterDataPipe, LineReader, Mapper, StreamReader


class FileLoggerSimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, logfile=None, **kwargs):
        self.__loggerHandle = None
        if logfile is not None:
            self.__loggerHandle = open(logfile, "a+")
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        if self.__loggerHandle is not None:
            self.__loggerHandle.write(f"{self.address_string()} - - [{self.log_date_time_string()}] {format % args}\n")
        return

    def finish(self):
        if self.__loggerHandle is not None:
            self.__loggerHandle.close()
        super().finish()


def set_up_local_server_in_thread():
    try:
        Handler = partial(FileLoggerSimpleHTTPRequestHandler, logfile=None)
        socketserver.TCPServer.allow_reuse_address = True

        server = socketserver.TCPServer(("", 0), Handler)
        server_addr = f"{server.server_address[0]}:{server.server_address[1]}"
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()

        # Wait a bit for the server to come up
        time.sleep(3)

        return server_thread, server_addr, server
    except Exception:
        raise


def create_temp_files_for_serving(tmp_dir, file_count, file_size, file_url_template):
    furl_local_file = os.path.join(tmp_dir, "urls_list")
    f = os.path.join(tmp_dir, "webfile_test_0.data")  # One generated file will be read repeatedly
    write_chunk = 1024 * 1024 * 16
    rmn_size = file_size
    with open(f, "ab+") as fout:
        while rmn_size > 0:
            fout.write(os.urandom(min(rmn_size, write_chunk)))
            rmn_size = rmn_size - min(rmn_size, write_chunk)
    with open(furl_local_file, "w") as fsum:
        for _ in range(file_count):
            fsum.write(file_url_template.format(num=0))


class TestHttpStress(expecttest.TestCase):
    __server_thread: threading.Thread
    __server_addr: str
    __server: socketserver.TCPServer

    @classmethod
    def setUpClass(cls):
        try:
            (cls.__server_thread, cls.__server_addr, cls.__server) = set_up_local_server_in_thread()
        except Exception as e:
            warnings.warn(
                "TestHttpStress could\
                          not set up due to {}".format(
                    str(e)
                )
            )

    @classmethod
    def tearDownClass(cls):
        try:
            cls.__server.shutdown()
            cls.__server_thread.join(timeout=15)
        except Exception as e:
            warnings.warn(
                "TestHttpStress could\
                           not tear down (clean up temp directory or terminate\
                           local server) due to {}".format(
                    str(e)
                )
            )

    def _http_test_base(self, test_file_size, test_file_count, timeout=None, chunk=None):
        def _get_data_from_tuple_fn(data):
            return data[1]

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
            # create tmp dir and files for test
            base_tmp_dir = os.path.basename(os.path.normpath(tmpdir))
            url = "http://{server_addr}/{tmp_dir}/webfile_test_{num}.data\n"
            file_url_template = url.format(server_addr=self.__server_addr, tmp_dir=base_tmp_dir, num="{num}")
            create_temp_files_for_serving(tmpdir, test_file_count, test_file_size, file_url_template)

            datapipe_dir_f = FileLister(tmpdir, "*_list")
            datapipe_stream = FileOpener(datapipe_dir_f, mode="r")
            datapipe_f_lines = LineReader(datapipe_stream)  # type: ignore[arg-type]
            datapipe_line_url: IterDataPipe[str] = Mapper(datapipe_f_lines, _get_data_from_tuple_fn)
            datapipe_http = HttpReader(datapipe_line_url, timeout=timeout)
            datapipe_tob = StreamReader(datapipe_http, chunk=chunk)

            for (url, data) in datapipe_tob:
                self.assertGreater(len(url), 0)
                self.assertRegex(url, r"^http://.+\d+.data$")
                if chunk is not None:
                    self.assertEqual(len(data), chunk)
                else:
                    self.assertEqual(len(data), test_file_size)

    @slowTest
    def test_stress_http_reader_iterable_datapipes(self):
        test_file_size = 1024
        test_file_count = 1024 * 16
        self._http_test_base(test_file_size, test_file_count)

    @slowTest
    def test_large_files_http_reader_iterable_datapipes(self):
        test_file_size = 1024 * 1024 * 128
        test_file_count = 200
        timeout = 30
        chunk = 1024 * 1024 * 8
        self._http_test_base(test_file_size, test_file_count, timeout=timeout, chunk=chunk)


if __name__ == "__main__":
    unittest.main()
