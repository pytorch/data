# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fnmatch import fnmatch
from typing import Dict, Iterator, Tuple, Union

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("extract_keys")
class ExtractKeysIterDataPipe(IterDataPipe[Dict]):
    r"""
    Given a stream of dictionaries, return a stream of tuples by selecting keys using glob patterns.

    Args:
        source_datapipe: a DataPipe yielding a stream of dictionaries.
        duplicate_is_error: it is an error if the same key is selected twice (True)
        ignore_missing: skip any dictionaries where one or more patterns don't match (False)
        *args: list of glob patterns or list of glob patterns
        duplicate_is_error: it is an error if the same key is selected twice (True)
        ignore_missing: allow patterns not to match (i.e., incomplete outputs)
        as_tuple: return a tuple instead of a dictionary

    Returns:
        a DataPipe yielding a stream of tuples

    Examples:
        >>> dp = FileLister(...).load_from_tar().webdataset().decode(...).extract_keys(["*.jpg", "*.png"], "*.gt.txt")
    """

    def __init__(
        self, source_datapipe: IterDataPipe[Dict], *args, duplicate_is_error=True, ignore_missing=False, as_tuple=False
    ) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[Dict] = source_datapipe
        self.duplicate_is_error = duplicate_is_error
        self.patterns = args
        self.ignore_missing = ignore_missing
        self.as_tuple = as_tuple

    def __iter__(self) -> Union[Iterator[Tuple], Iterator[Dict]]:  # type: ignore
        for sample in self.source_datapipe:
            result = []
            used = set()
            for pattern in self.patterns:
                pattern = [pattern] if not isinstance(pattern, (list, tuple)) else pattern
                matches = [x for x in sample.keys() if any(fnmatch(x, p) for p in pattern)]
                if len(matches) == 0:
                    if self.ignore_missing:
                        continue
                    else:
                        raise ValueError(f"extract_keys: cannot find {pattern} in sample keys {sample.keys()}.")
                if len(matches) > 1 and self.duplicate_is_error:
                    raise ValueError(f"extract_keys: multiple sample keys {sample.keys()} match {pattern}.")
                if matches[0] in used and self.duplicate_is_error:
                    raise ValueError(f"extract_keys: key {matches[0]} is selected twice.")
                used.add(matches[0])
                value = sample[matches[0]]
                if self.as_tuple:
                    result.append(value)
                else:
                    result.append((matches[0], value))
            if self.as_tuple:
                yield tuple(result)
            else:
                yield {k: v for k, v in result}

    def __len__(self) -> int:
        return len(self.source_datapipe)
