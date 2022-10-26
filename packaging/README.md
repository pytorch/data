# Build TorchData Release

These are a collection of scripts that are to be used for release activities.

## Conda

### Release

```bash
PYTHON_VERSION=3.9 PYTORCH_VERSION=1.11.0 packaging/build_conda.sh
```

### Nightly

```bash
PYTHON_VERSION=3.9 packaging/build_conda.sh
```

## Wheel

### Release

```bash
PYTHON_VERSION=3.9 PYTORCH_VERSION=1.11.0 packaging/build_wheel.sh
```

### Nightly

```bash
PYTHON_VERSION=3.9 packaging/build_wheel.sh
```
## [`AWSSDK`](https://github.com/aws/aws-sdk-cpp)

The following table is the corresponding `torchdata` binaries with pre-compiled `AWSSDK` extension on different operating systems.

| `torchdata`        | `Wheel`            | `Conda`            |
| ------------------ | ------------------ | ------------------ |
| Linux              | :heavy_check_mark: | :heavy_check_mark: |
| Windows            | :heavy_check_mark: | :x:                |
| MacOS (x86_64)     | :heavy_check_mark: | :heavy_check_mark: |
| MacOS (arm64)      | :heavy_check_mark: | :heavy_check_mark: |

### Manylinux

`AWSSDK` requires OpenSSL and cURL. In order to provide `manylinux2014_x86_64` wheels with `AWSSDK` enabled, `torchdata` distributions are bundled with OpenSSL(1.1.1o) and cURL(7.38.1). If anything is out of date, please open an issue to request upgrading them.
