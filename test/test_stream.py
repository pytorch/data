import time

import expecttest
import os
import socketserver
import tarfile
import tempfile
import time
import threading
import unittest


from http.server import SimpleHTTPRequestHandler
from torch.utils.data.datapipes.iter import StreamReader
from torchdata.datapipes.iter import HttpReader, IterableWrapper
from typing import List, Tuple


# HTTP server setup
# Allows the server to start from an arbitrary directory, such as a TempDirectory
def handler_from(directory, tar_file_name, chunk_size, number_of_sends):
    def _init(self, *args, **kwargs):
        return SimpleHTTPRequestHandler.__init__(
            self, *args, directory=self.directory, **kwargs
        )  # type: ignore[call-arg]

    def _do_GET(self) -> None:
        self.send_response(200)
        # self.send_header("Content-Type", "application/x-tar")
        # self.send_header("User-Information", "Sending TAR file stream")
        self.end_headers()
        print(f"Going to send the files {number_of_sends} times")
        for i in range(number_of_sends):
            print(f"{i}th time of sending TAR file:")
            with open(tar_file_name, 'rb') as f:
                j = 0
                while True:
                    file_data = f.read(chunk_size)
                    if file_data is None or len(file_data) == 0:
                        break
                    print(f"Sending the {j}th chunk...")
                    self.wfile.write(file_data)
                    j += 1
                # Might need to slow down using a timer (it might automatically block me)
                # It might throw an exception, if so, sleep

    return type(f"HandlerFrom<{directory}>",
                (SimpleHTTPRequestHandler,),
                {"__init__": _init, "directory": directory, "do_GET": _do_GET},)

class TestStream(expecttest.TestCase):
    def setUp(self) -> None:
        # Temporary Directory abd Server Setup
        print("\nSetup is running...")
        self.temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
        self.temp_dir_path = self.temp_dir.name
        self.port = 8004

    def tearDown(self) -> None:
        print("Tear down class is running...")
        if self.server:
            self.stop_server = True
            self.server_thread.join(timeout=3)
            self.server.shutdown()
            print("Server is stopped")
        print("Removing Temp Directory...")
        self.temp_dir.cleanup()
        print("Tear down has completed")

    # TODO: Should this be test dependent? Refactor this to take in arguments?
    def running_server(self):
        httpd = socketserver.TCPServer(("", self.port),
                                       handler_from(self.temp_dir_path,
                                                    self.tar_file_name,
                                                    self.server_chunk_size,
                                                    self.number_of_sends))
        print(f"Starting TCP server on port {self.port}...")
        self.server = httpd
        httpd.serve_forever()
        while True:
            if self.stop_server:
                httpd.shutdown()
                httpd.server_close()
                break

    @staticmethod
    def generate_binary_files(path, num_files: int, num_kbs: int = 16) -> List[str]:
        print("Generating binary files...")
        ls = []
        for i in range(num_files):
            with tempfile.NamedTemporaryFile(dir=path, delete=False, prefix=f"{i}", suffix=".byte") as f:
                temp_binary_file_name = os.path.join(path, f.name)
            with open(temp_binary_file_name, "wb") as binary_file:
                binary_file.write(b"0123456789abcdef" * 64 * num_kbs)
            ls.append(temp_binary_file_name)
        return ls

    @staticmethod
    def get_name_mode_from_file_type(file_type: str) -> Tuple[str, str, str]:
        if file_type == "tar":  # Uncompressed
            return "test_tar.tar", "w:tar", "r|tar"  # Must use '|' instead of ':' to get a stream
        elif file_type == "gz":
            return "test_tar.tar.gz", "w:gz", "r|gz"
        elif file_type == "xz":
            return "test_tar.tar.xz", "w:xz", "r|xz"
        else:
            raise ValueError(f"file_type must be 'tar', 'gz', or 'xz', but instead found {file_type}.")

    @staticmethod
    def add_to_tar_archive(file_path, write_mode: str, file_names: List[str]) -> None:
        print("Adding files to Tar archive...")
        with tarfile.open(file_path, write_mode) as tar:
            for fname in file_names:
                tar.add(fname, arcname=os.path.basename(fname))
        print(f"The path to the file is: {file_path}.")
        print(f"The file size is {os.path.getsize(file_path)} bytes.")

    def test_tar_stream(self) -> None:
        os.chdir(self.temp_dir_path)
        # Within the generated tar: 5 files, 1MB each
        print("Changed curr path...")
        # TODO: Doesn't work when num_kbs > 16
        file_paths = TestStream.generate_binary_files(path=self.temp_dir_path, num_files=5, num_kbs=4)
        file_type = "tar"
        tar_file_name, write_mode, read_mode = TestStream.get_name_mode_from_file_type(file_type=file_type)
        self.tar_file_name = tar_file_name
        self.server_chunk_size = 1024 * 16  # Number of bytes the server will send at a time
        self.number_of_sends = 5  # The server will send the same set of files 10 times
        temp_tarfile_pathname = os.path.join(self.temp_dir_path, tar_file_name)
        TestStream.add_to_tar_archive(temp_tarfile_pathname, write_mode, file_paths)  # Save files to archive

        self.stop_server = False
        self.server_thread = threading.Thread(target=self.running_server)
        self.server_thread.start()


        file_url = f"""http://localhost:{self.port}/{tar_file_name}"""
        print(f"Opening file_url: {file_url}")

        http_reader_dp = HttpReader(IterableWrapper([file_url]))
        tar_dp = http_reader_dp.read_from_tar(mode=read_mode)
        # stream_read_dp = StreamReader(tar_dp, chunk=1024)
        # TODO: Need to verify if the reader is prematurely ending
        for _fname, stream in tar_dp:
            while True:
                chunk = stream.read(1024)
                if chunk:
                    print(chunk)
                # self.assertEqual(b"0123456789abcdef", chunk)
                if not chunk:
                    break


if __name__ == "__main__":
    unittest.main()
