# S3 IO Datapipe Documentation

## Build from Source

`ninja` is required to link PyThon implementation to C++ source code.

```bash
conda install ninja
```

S3 IO datapipes are included when building with flag `BUILD_S3=1`. The following commands can build `torchdata` from
source with S3 datapipes.

```bash
BUILD_S3=1 python setup.py install
```

We also offer nightly and official (>=0.4.0) TorchData releases integrated with `AWSSDK` on the most of platforms.
Please check the [link](https://github.com/pytorch/data/tree/main/packaging#awssdk) for the list of supported platforms
with the pre-assembled binaries.

If you'd like to use customized installations of `pybind11` or `aws-sdk-cpp`, you may set the following flags when
building from source.

```
USE_SYSTEM_PYBIND11=1
USE_SYSTEM_AWS_SDK_CPP=1
USE_SYSTEM_LIBS=1 # uses both pre-installed pybind11 and aws-sdk-cpp
```

Note: refer to the official documentation for detailed installtion instructions of
[aws-sdk-cpp](https://github.com/aws/aws-sdk-cpp).

### Example

Please refer to the documentation:

- [`S3FileLister`](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.S3FileLister.html#s3filelister)
- [`S3FileLister`](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.S3FileLoader.html#s3fileloader)

### Note

It's recommended to set up a detailed configuration file with the `AWS_CONFIG_FILE` environment variable. The following
environment variables are also parsed: `HOME`, `S3_USE_HTTPS`, `S3_VERIFY_SSL`, `S3_ENDPOINT_URL`, `AWS_REGION` (would
be overwritten by the `region` variable).

## Troubleshooting

If you get `Access Denied`, it's very possibly a
[wrong region configuration](https://github.com/aws/aws-sdk-cpp/issues/1211) or an
[accessing issue with `aws-sdk-cpp`](https://aws.amazon.com/premiumsupport/knowledge-center/s3-access-denied-aws-sdk/).
