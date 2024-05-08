# Iterable Datapipes

## S3 IO Datapipe Documentation

**WARNING**: S3 IO Datapipes have been deprecated. Consider using [S3 Connector for PyTorch](https://github.com/awslabs/s3-connector-for-pytorch).

### Build from Source

`ninja` is required to link PyThon implementation to C++ source code.

```bash
conda install ninja
```

S3 IO datapipes are included when building with flag `BUILD_S3=1`. The following commands can build `torchdata` from
source with S3 datapipes.

```bash
BUILD_S3=1 pip install .
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
- [`S3FileLoader`](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.S3FileLoader.html#s3fileloader)

### Note

Your environment must be properly configured for AWS to use the DataPipes. It is possible to do that via the AWS Command
Line Interface (`aws configure`).

It's recommended to set up a detailed configuration file with the `AWS_CONFIG_FILE` environment variable. The following
environment variables are also parsed: `HOME`, `S3_USE_HTTPS`, `S3_VERIFY_SSL`, `S3_ENDPOINT_URL`, `AWS_REGION` (would
be overwritten by the `region` variable).

### Troubleshooting

If you get `Access Denied` or no response, it's very possibly a
[wrong region configuration](https://github.com/aws/aws-sdk-cpp/issues/1211) or an
[accessing issue with `aws-sdk-cpp`](https://aws.amazon.com/premiumsupport/knowledge-center/s3-access-denied-aws-sdk/).

## AIStore IO Datapipe

[AIStore](https://github.com/NVIDIA/aistore) (AIS for short) is a highly available lightweight object storage system
that specifically focuses on petascale deep learning. As a reliable redundant storage, AIS supports n-way mirroring and
erasure coding. But it is not purely – or not only – a storage system: it’ll shuffle user datasets and run custom
extract-transform-load workloads.

AIS is an elastic cluster that can grow and shrink at runtime and can be ad-hoc deployed, with or without Kubernetes,
anywhere from a single Linux machine to a bare-metal cluster of any size.

AIS fully supports Amazon S3, Google Cloud, and Microsoft Azure backends, providing a unified namespace across multiple
connected backends and/or other AIS clusters, and [more](https://github.com/NVIDIA/aistore#features). Getting started
with AIS will take only a few minutes (prerequisites boil down to having a Linux with a disk) and can be done either by
running a prebuilt all-in-one docker image or directly from the open-source.

### Dependency

The `AISFileLister` and `AISFileLoader` under [`aisio.py`](/torchdata/datapipes/iter/load/aisio.py) internally use the
[Python SDK](https://github.com/NVIDIA/aistore/tree/master/sdk/python) for AIStore.

Run `pip install aistore` or `conda install aistore` to install the [python package](https://pypi.org/project/aistore/).

### Example

Please refer to the documentation:

- [`AISFileLister`](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.AISFileLister.html#aisfilelister)
- [`AISFileLoader`](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.AISFileLoader.html#aisfileloader)
