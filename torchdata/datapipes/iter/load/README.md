# S3 IO Datapipe Documentation

## Installation

Torchdata S3 IO datapipes depends on [aws-sdk-cpp](https://github.com/aws/aws-sdk-cpp). The following is just a
recommended way to installing aws-sdk-cpp, please refer to official documentation for detailed instructions.

```bash
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp
cd aws-sdk-cpp/
mkdir sdk-build
cd sdk-build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3;transfer"
make
make install # may need sudo
```

`ninja` and `pybind11` are also required to link PyThon implementation to C++ source code.

```bash
conda install ninja pybind11
```

S3 IO datapipes are't included when building by default. To build S3 IO in `torchdata`, at the `/data` root folder, run
the following commands.

```bash
export BUILD_S3=ON
pip uninstall torchdata -y
python setup.py clean
python setup.py install
```

## Using S3 IO datapies

### S3FileLister

`S3FileLister` accepts a list of S3 prefixes and iterates all matching s3 urls. The functional API is `list_file_by_s3`.
Acceptable prefixes include `s3://bucket-name`, `s3://bucket-name/`, `s3://bucket-name/folder`,
`s3://bucket-name/folder/`, and `s3://bucket-name/prefix`. You may also set `length`, `request_timeout_ms` (default 3000
ms in aws-sdk-cpp), and `region`. Note that:

1. Input **must** be a list and direct S3 URLs are skipped.
2. `length` is `-1` by default, and any call to `__len__()` is invalid, because the length is unknown until all files
   are iterated.
3. `request_timeout_ms` and `region` will overwrite settings in the configuration file or environment variables.

### S3FileLoader

`S3FileLoader` accepts a list of S3 URLs and iterates all files in `BytesIO` format with `(url, BytesIO)` tuples. The
functional API is `load_file_by_s3`. You may also set `request_timeout_ms` (default 3000 ms in aws-sdk-cpp), `region`,
`buffer_size` (default 120Mb), and `multi_part_download` (default to use multi-part downloading). Note that:

1. Input **must** be a list and S3 URLs must be valid.
2. `request_timeout_ms` and `region` will overwrite settings in the configuration file or environment variables.

### Example

```py
from torchdata.datapipes.iter import S3FileLister, S3FileLoader

s3_prefixes = ['s3://bucket-name/folder/', ...]
dp_s3_urls = S3FileLister(s3_prefixes)
dp_s3_files = S3FileLoader(s3_urls) # outputs in (url, StreamWrapper(BytesIO))
# more datapipes to convert loaded bytes, e.g.
datapipe = StreamWrapper(dp_s3_files).parse_csv(delimiter=' ')

for d in datapipe: # Start loading data
    pass
```

### Note

It's recommended to set up a detailed configuration file with the `AWS_CONFIG_FILE` environment variable. The following
environment variables are also parsed: `HOME`, `S3_USE_HTTPS`, `S3_VERIFY_SSL`, `S3_ENDPOINT_URL`, `AWS_REGION` (would
be overwritten by the `region` variable).
