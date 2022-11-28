# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import warnings
from functools import partial
from io import BufferedIOBase
from typing import Any, cast, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import torch

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from torchdata.datapipes.utils.common import validate_pathname_binary_tuple

try:
    from math import prod  # type: ignore
except ImportError:
    # Implementation for older Python
    # NOTE: this is not supported by mypy yet
    # https://github.com/python/mypy/issues/1393
    import operator
    from functools import reduce

    def prod(xs):  # type: ignore[no-redef]
        return reduce(operator.mul, xs, 1)


try:
    import google.protobuf as _protobuf

    del _protobuf
    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False

U = Union[bytes, bytearray, str]
TFRecordFeatureSpec = Tuple[Tuple[int, ...], torch.dtype]
TFRecordExampleSpec = Dict[str, TFRecordFeatureSpec]

#  Note, reccursive types not supported by mypy at the moment
# TODO(640): uncomment as soon as it becomes supported
#  https://github.com/python/mypy/issues/731
#  BinaryData = Union[str, List['BinaryData']]
TFRecordBinaryData = Union[str, List[str], List[List[str]], List[List[List[Any]]]]
TFRecordExampleFeature = Union[torch.Tensor, List[torch.Tensor], TFRecordBinaryData]
TFRecordExample = Dict[str, TFRecordExampleFeature]


class SequenceExampleSpec(NamedTuple):
    context: TFRecordExampleSpec
    feature_lists: TFRecordExampleSpec


def _assert_protobuf() -> None:
    if not HAS_PROTOBUF:
        raise ModuleNotFoundError(
            "Package `protobuf` is required to be installed to use this datapipe."
            "Please use `pip install protobuf` or `conda install -c conda-forge protobuf`"
            "to install the package"
        )


def iterate_tfrecord_file(data: BufferedIOBase) -> Iterator[memoryview]:
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    data_bytes = bytearray(1024)

    while True:
        bytes_read = data.readinto(length_bytes)
        if bytes_read == 0:
            break
        elif bytes_read != 8:
            raise RuntimeError("Invalid tfrecord file: failed to read the record size.")
        if data.readinto(crc_bytes) != 4:
            raise RuntimeError("Invalid tfrecord file: failed to read the start token.")
        (length,) = struct.unpack("<Q", length_bytes)
        if length > len(data_bytes):
            data_bytes = data_bytes.zfill(int(length * 1.5))
        data_bytes_view = memoryview(data_bytes)[:length]
        if data.readinto(data_bytes_view) != length:
            raise RuntimeError("Invalid tfrecord file: failed to read the record.")
        if data.readinto(crc_bytes) != 4:
            raise RuntimeError("Invalid tfrecord file: failed to read the end token.")

        # TODO(641): check CRC
        yield data_bytes_view


def process_feature(feature) -> torch.Tensor:
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]
    inferred_typename, value = field[0].name, field[1].value
    if inferred_typename == "bytes_list":
        pass
    elif inferred_typename == "float_list":
        value = torch.tensor(value, dtype=torch.float32)
    elif inferred_typename == "int64_list":
        value = torch.tensor(value, dtype=torch.int64)
    return value


def _reshape_list(value, shape):
    # Flatten list
    flat_list = []

    def flatten(value):
        if isinstance(value, (str, bytes)):
            flat_list.append(value)
        else:
            for x in value:
                flatten(x)

    flatten(value)

    # Compute correct shape
    common_divisor = prod(x for x in shape if x != -1)
    if sum(1 for x in shape if x == -1) > 1:
        raise RuntimeError("Shape can contain at most one dynamic dimension (-1).")
    if len(flat_list) % max(common_divisor, 1) != 0:
        raise RuntimeError(f"Cannot reshape {len(flat_list)} values into shape {shape}")
    shape = [x if x != -1 else (len(flat_list) // common_divisor) for x in shape]

    # Reshape list into the correct shape
    def _reshape(value, shape):
        if len(shape) == 0:
            assert len(value) == 1
            return value[0]
        elif len(shape) == 1:  # To make the reccursion faster
            assert len(value) == shape[0]
            return value
        dim_size = len(value) // shape[0]
        return [_reshape(value[i * dim_size : (i + 1) * dim_size], shape[1:]) for i in range(dim_size)]

    return _reshape(flat_list, shape)


def _apply_feature_spec(value, feature_spec):
    if feature_spec is not None:
        shape, dtype = feature_spec
        if isinstance(dtype, torch.dtype):
            if shape is not None:
                value = value.reshape(shape)
            value = value.to(dtype)
        elif shape is not None:
            # Manual list reshape
            value = _reshape_list(value, shape)
    return value


def _parse_tfrecord_features(features, spec: Optional[TFRecordExampleSpec]) -> Dict[str, torch.Tensor]:
    result = dict()
    features = features.feature
    for key in features.keys():
        if spec is not None and key not in spec:
            continue
        feature_spec = None if spec is None else spec[key]
        feature = features[key]
        result[key] = _apply_feature_spec(process_feature(feature), feature_spec)
    return result


def parse_tfrecord_sequence_example(example, spec: Optional[TFRecordExampleSpec]) -> TFRecordExample:
    # Parse context features
    result = cast(TFRecordExample, _parse_tfrecord_features(example.context, spec))

    # Parse feature lists
    feature_lists_keys = None if spec is None else set(spec.keys()) - set(result.keys())
    features = example.feature_lists.feature_list
    for key in features.keys():
        if feature_lists_keys is not None and key not in feature_lists_keys:
            continue
        feature_spec = None if spec is None else spec[key]
        feature = features[key].feature
        if key in result:
            raise RuntimeError(
                "TFRecord example's key {key} is contained in both the context and feature lists. This is not supported."
            )

        value: Union[torch.Tensor, List[Any]] = list(map(partial(process_feature), feature))

        # For known torch dtypes, we stack the list features
        if feature_spec is not None and isinstance(feature_spec[1], torch.dtype):
            value = torch.stack(cast(List[torch.Tensor], value), 0)
        value = _apply_feature_spec(value, feature_spec)
        result[key] = value
    if spec is not None and len(result.keys()) != len(spec.keys()):
        raise RuntimeError(f"Example is missing some required keys: {sorted(result.keys())} != {sorted(spec.keys())}")
    return result


@functional_datapipe("load_from_tfrecord")
class TFRecordLoaderIterDataPipe(IterDataPipe[TFRecordExample]):
    r"""
    Opens/decompresses tfrecord binary streams from an Iterable DataPipe which contains tuples of path name and
    tfrecord binary stream, and yields the stored records (functional name: ``load_from_tfrecord``).

    Args:
        datapipe: Iterable DataPipe that provides tuples of path name and tfrecord binary stream
        length: a nominal length of the DataPipe

    Note:
        The opened file handles will be closed automatically if the default ``DecoderDataPipe``
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.

    Example:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener
        >>> datapipe1 = FileLister(".", "*.tfrecord")
        >>> datapipe2 = FileOpener(datapipe1, mode="b")
        >>> tfrecord_loader_dp = datapipe2.load_from_tfrecord()
        >>> for example in tfrecord_loader_dp:
        >>>     print(example)
    """

    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        spec: Optional[TFRecordExampleSpec] = None,
        length: int = -1,
    ) -> None:
        super().__init__()
        _assert_protobuf()

        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length: int = length
        self.spec = spec

    def __iter__(self) -> Iterator[TFRecordExample]:
        # We assume that the "example.proto" and "feature.proto"
        # stays the same for future TensorFlow versions.
        # If it changed, newer TensorFlow versions would
        # not be able to load older tfrecord datasets.
        from .protobuf_template import _tfrecord_example_pb2 as example_pb2

        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                for example_bytes in iterate_tfrecord_file(data_stream):
                    example = example_pb2.SequenceExample()  # type: ignore
                    example.ParseFromString(example_bytes)  # type: ignore
                    yield parse_tfrecord_sequence_example(example, self.spec)
            except RuntimeError as e:
                warnings.warn(f"Unable to read from corrupted tfrecord stream {pathname} due to: {e}, abort!")
                raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
