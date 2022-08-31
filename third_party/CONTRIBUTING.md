# Third-Party Libraries

The `third_party` directory contains all of the (optional) dependency libraries used by `torchdata`. And, it relies on
the `CMake` system to compile and integrate them into a C-extension module that can be found as `torchdata/_torchdata`.

`torchdata` also relies on [`pybind11`](https://github.com/pybind/pybind11) to expose C++ API in Python. Please refer
this [directory](https://github.com/pytorch/data/tree/main/torchdata/csrc) for more detail.

## Integration Requirement

### Soft Dependency vs Hard Dependency

`TorchData` as a data-processing libraries provides a bunch of `DataPipes` that are integrated with different
third-party libraries to handle specific use cases. For example, `datasets` is imported to load dataset from
`HuggingFace`, and `fsspec` is imported to provide a unified API to access and load from local or remote file systems.
Those dependencies are optional and should only be verified of their availability when they are used/referenced by users
during `DataPipe` construction. You can find examples from
[here](https://github.com/pytorch/data/blob/bb78231e5f87620385cb2f91cda87e7f9414eb4a/torchdata/datapipes/iter/load/huggingface.py#L57-L62)
and
[here](https://github.com/pytorch/data/blob/d19858202df7e8b75765074259e6023f539cbf3f/torchdata/datapipes/iter/load/fsspec.py#L59).
They are contrasted with Core dependencies that are must be installed along with `torchdata`.

- Optional features
  - For a Python library, please follow the example above to add soft dependency to `torchdata`.
  - For a C library, a compilation flag should be provided to users to enable or disable compilation and integration via
    `CMake`. For example,
    [`BUILD_S3`](https://github.com/pytorch/data/blob/87d6dc3d6b0df6829cc2813a0ca033accfa9d795/torchdata/csrc/CMakeLists.txt#L7)
    is provided for `AWSSDK`.
- Core features
  - Add Python library as a hard dependency to
    [`requirements.txt`](https://github.com/pytorch/data/blob/main/requirements.txt).
  - Add C library as a submodule under the `third_party` and compile it against `torchdata` C-extension as always.

### Static Linking vs Dynamic Linking

For the third-party libraries in C, if their runtime libraries are available on both PyPI and Conda across platfroms
(Linux, MacOS, Windows and Python 3.7-3.10), it's recommended to use dynamic linking against the `torchdata`
C-extension.

For the third-party libraries that are not available on PyPI and Conda, please add it as a submodule under `third_party`
directory and statically compile it with the `torchdata` C-extension.

Notes:

- On `Linux` OS, static libraries are required to follow the `manylinux2014` protocol (equivalent of `manylinux_2_17`)
  when they are integrated with `torchdata`.
