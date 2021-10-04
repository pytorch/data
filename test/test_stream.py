import http.server
import os
import socketserver
import tarfile
import tempfile
import threading

from torchdata.datapipes.iter import HttpReader, IterableWrapper, TarArchiveReader


# Tempory Directory and FilesSetup
temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
temp_dir_path = temp_dir.name

os.chdir(temp_dir_path)

with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, prefix="1", suffix=".txt") as f:
    temp_file1_name = os.path.basename(f.name)
with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, prefix="2", suffix=".byte") as f:
    temp_file2_name = os.path.basename(f.name)

with open(temp_file1_name, "w") as f1:
    f1.write("0123456789abcdef")
with open(temp_file2_name, "wb") as f2:
    f2.write(b"0123456789abcdef")

tar_file_name = "test_tar.tar.gz"

# Creating the .tar file
# TODO: We want to be able to create a .tar file of arbitarily large size
temp_tarfile_pathname = os.path.join(temp_dir.name, tar_file_name)
with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
    tar.add(temp_file1_name)
    tar.add(temp_file2_name)

# HTTP server setup
class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=temp_dir_path, **kwargs)

port = 8005
httpd = socketserver.TCPServer(("", port), Handler)

print(f"Starting server on port {port}:")
threading.Thread(target=httpd.serve_forever).start()

file_url = f"""http://localhost:8005/{tar_file_name}"""

try:
    print(f"Opening file_url: {file_url}")

    # Using cache works
    root = temp_dir.name
    cache_dp = IterableWrapper([file_url]).on_disk_cache(
        HttpReader,
        op_map=lambda x: (x[0], x[1].read()),
        filepath_fn=lambda x: os.path.join(root, tar_file_name),
    )
    tar_dp = cache_dp.read_from_tar()

    # This doesn't work, because tarfile.open needs the ability to seek from the stream
    # http_reader_dp = HttpReader(IterableWrapper([file_url]))
    # tar_dp = http_reader_dp.read_from_tar()

    # Print result
    for fname, stream in tar_dp:
        print(f"{fname}: {stream.read()}")
finally:
    httpd.server_close()