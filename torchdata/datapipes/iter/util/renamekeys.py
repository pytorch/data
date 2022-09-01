# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from fnmatch import fnmatch
from typing import Dict, Iterator, List, Union, Any

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe("rename_keys")
class KeyRenamerIterDataPipe(IterDataPipe[Dict]):
    r"""
    Given a stream of dictionaries, rename keys using glob patterns.

    This is used for quickly extracting relevant fields from a stream of dictionaries
    and renaming them to a common format.

    Note that if keys contain slashes, only the part after the last slash is matched.

    Args:
        source_datapipe: a DataPipe yielding a stream of dictionaries.
        keep_unselected: keep keys/value pairs even if they don't match any pattern (False)
        must_match: all key value pairs must match (True)
        duplicate_is_error: it is an error if two renamings yield the same key (True)
        *args: `(renamed, pattern)` pairs
        **kw: `renamed=pattern` pairs

    Returns:
        a DataPipe yielding a stream of dictionaries.

    Examples:
        >>> dp = IterableWrapper([{"/a/b.jpg": b"data"}]).rename_keys(("image", "*.jpg"))
        >>> list(dp)
        [{'image': b'data'}]
        >>> dp = IterableWrapper([
            {"/a/b.input.jpg": b"data1", "/a/b.target.jpg": b"data2"},
            {"/a/b.input.png": b"data1", "/a/b.target.png": b"data2"},
        ]).rename_keys(input="*.input.*", output="*.target.*")
        >>> list(dp)
        [{'input': b'data1', 'target': b'data2'}, {'input': b'data1', 'target': b'data2'}]
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[Dict[Any, Any]],
        *args,
        keep_unselected=False,
        must_match=True,
        duplicate_is_error=True,
        **kw,
    ) -> None:
        super().__init__()
        assert not (keep_unselected and must_match)
        self.source_datapipe: IterDataPipe[List[Union[Dict, List]]] = source_datapipe
        self.must_match = must_match
        self.keep_unselected = keep_unselected
        self.duplicate_is_error = duplicate_is_error
        self.renamings = [(pattern, output) for output, pattern in args]
        self.renamings += [(pattern, output) for output, pattern in kw.items()]

    def __iter__(self) -> Iterator[Dict]:
        for sample in self.source_datapipe:
            new_sample = {}
            matched = {k: False for k, _ in self.renamings}
            for path, value in sample.items():
                fname = re.sub(r".*/", "", path)
                new_name = None
                for pattern, name in self.renamings[::-1]:
                    if fnmatch(fname.lower(), pattern):
                        matched[pattern] = True
                        new_name = name
                        break
                if new_name is None:
                    if self.keep_unselected:
                        new_sample[path] = value
                    continue
                if new_name in new_sample:
                    if self.duplicate_is_error:
                        raise ValueError(f"Duplicate value in sample {sample.keys()} after rename.")
                    continue
                new_sample[new_name] = value
            if self.must_match and not all(matched.values()):
                raise ValueError(f"Not all patterns ({matched}) matched sample keys ({sample.keys()}).")

            yield new_sample

    def __len__(self) -> int:
        return len(self.source_datapipe)
