import expecttest
import http.server
import os
import socketserver
import tarfile
import tempfile
import threading
import unittest


from torch.utils.data.datapipes.iter import StreamReader
from torchdata.datapipes.iter import HttpReader, IterableWrapper, TarArchiveReader, GDriveReader, OnlineReader
from typing import IO, Iterable, Iterator, Optional, Tuple, cast, List


# HTTP server setup
# Allows the server to start from an arbitrary directory, such as a TempDirectory
def handler_from(directory):
    def _init(self, *args, **kwargs):
        return http.server.SimpleHTTPRequestHandler.__init__(self, *args, directory=self.directory, **kwargs)

    return type(f'HandlerFrom<{directory}>',
                (http.server.SimpleHTTPRequestHandler,),
                {'__init__': _init, 'directory': directory})


class TestStream(expecttest.TestCase):

    def setUp(self) -> None:
        # Temporary Directory and FilesSetup
        print("\nSetup is running...")
        self.temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
        self.temp_dir_path = self.temp_dir.name
        self.port = 8006
        self.start_test_server(self.temp_dir_path, self.port)

    def tearDown(self) -> None:
        print("Tear down is running...")
        self.stop_test_server(self.port)

    def start_test_server(self, path, port: int) -> None:
        self.httpd = socketserver.TCPServer(("", port), handler_from(path))
        print(f"Starting TCP server on port {port}...")
        threading.Thread(target=self.httpd.serve_forever).start()

    def stop_test_server(self, port: int):
        self.httpd.server_close()  # TODO: Might need to call thread to close this?
        print(f"Stopping TCP server on port {port}...")


    @staticmethod
    def generate_binary_files(path, num_files: int, num_kbs: int = 16):
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
                tar.add(fname)
        print(f"The path to the file is: {file_path}.")
        print(f"The file size is {os.path.getsize(file_path)} bytes.")

    def test_tar_stream(self):
        os.chdir(self.temp_dir_path)
        file_names = TestStream.generate_binary_files(path=self.temp_dir_path, num_files=10, num_kbs=1024)
        file_type = "tar"
        tar_file_name, write_mode, read_mode = TestStream.get_name_mode_from_file_type(file_type=file_type)
        temp_tarfile_pathname = os.path.join(self.temp_dir_path, tar_file_name)
        TestStream.add_to_tar_archive(temp_tarfile_pathname, write_mode, file_names)  # Save files to archive

        file_url = f"""http://localhost:{self.port}/{tar_file_name}"""
        print(f"Opening file_url: {file_url}")

        http_reader_dp = HttpReader(IterableWrapper([file_url]))
        tar_dp = http_reader_dp.read_from_tar(mode=read_mode)
        stream_read_dp = StreamReader(tar_dp, chunk=16)

        # Result
        # for fname, stream in tar_dp:
        #     print(f"{fname}: {stream.read()}")
        for fname, chunk in stream_read_dp:
            print(f"{fname}: {chunk}")
            self.assertEqual(b"0123456789abcdef", chunk)


if __name__ == "__main__":
    unittest.main()
