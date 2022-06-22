# S3 IO Datapipe Documentation

## Build from Source

`ninja` is required to link PyThon implementation to C++ source code.

```bash
conda install ninja
```

S3 IO datapipes are included when building with flag `BUILD_S3=1`. The following commands can build `torchdata` from
source with S3 datapipes.

```bash
pip uninstall torchdata -y
git clone https://github.com/pytorch/data.git
cd data
python setup.py clean
BUILD_S3=1 python setup.py install
```

If you'd like to use customized installations of `pybind11` or `aws-sdk-cpp`, you may set the following flags when
building from source.

```
USE_SYSTEM_PYBIND11=1
USE_SYSTEM_AWS_SDK_CPP=1
USE_SYSTEM_LIBS=1 # uses both pre-installed pybind11 and aws-sdk-cpp
```

Note: refer to the official documentation for detailed installtion instructions of
[aws-sdk-cpp](https://github.com/aws/aws-sdk-cpp).

## Using S3 IO datapies

### S3FileLister

`S3FileLister` accepts a list of S3 prefixes and iterates all matching s3 urls. The functional API is
`list_files_by_s3`. Acceptable prefixes include `s3://bucket-name`, `s3://bucket-name/`, `s3://bucket-name/folder`,
`s3://bucket-name/folder/`, and `s3://bucket-name/prefix`. You may also set `length`, `request_timeout_ms` (default 3000
ms in aws-sdk-cpp), and `region`. Note that:

1. Input **must** be a list and direct S3 URLs are skipped.
2. `length` is `-1` by default, and any call to `__len__()` is invalid, because the length is unknown until all files
   are iterated.
3. `request_timeout_ms` and `region` will overwrite settings in the configuration file or environment variables.

### S3FileLoader

`S3FileLoader` accepts a list of S3 URLs and iterates all files in `BytesIO` format with `(url, BytesIO)` tuples. The
functional API is `load_files_by_s3`. You may also set `request_timeout_ms` (default 3000 ms in aws-sdk-cpp), `region`,
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

## Troubleshooting

If you get `Access Denied`, it's very possibly a
[wrong region configuration](https://github.com/aws/aws-sdk-cpp/issues/1211) or an
[accessing issue with `aws-sdk-cpp`](https://aws.amazon.com/premiumsupport/knowledge-center/s3-access-denied-aws-sdk/).
