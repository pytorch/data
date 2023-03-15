# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, final, Iterator, List, Tuple, TypeVar

from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


T_co = TypeVar("T_co", covariant=True)


def _default_line_join(lines: List[str]) -> str:
    return "\n".join(lines)


@functional_datapipe("lines_to_paragraphs")
class ParagraphAggregatorIterDataPipe(IterDataPipe[Tuple[str, str]]):
    r"""
    Aggregates lines of text from the same file into a single paragraph (functional name: ``lines_to_paragraphs``).
    Specifically, this accepts a DataPipe consisting of tuples of a file name and a line. For each tuple,
    it checks if the file name matches the file name from the previous tuple. If yes, it joins the current line
    with existing paragraph. If the file names do not match, the existing paragraph is yielded and a new
    paragraph starts.

    Args:
        source_datapipe: a DataPipe with tuples of a file name and a line
        joiner: a function that joins a list of lines together

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(
        >>>                 [("file1", "Line1"), ("file1", "Line2"), ("file2", "Line2,1"), ("file2", "Line2,2"), ("file2", "Line2,3")]
        >>>             )
        >>> para_agg_dp = source_dp.lines_to_paragraphs(joiner=lambda ls: " ".join(ls))
        >>> list(para_agg_dp)
        [('file1', 'Line1 Line2'), ('file2', 'Line2,1 Line2,2 Line2,3')]
    """

    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, T_co]], joiner: Callable = _default_line_join) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, T_co]] = source_datapipe
        _check_unpickable_fn(joiner)
        self.joiner: Callable = joiner
        self.buffer: List = []

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        prev_filename = None
        for filename, line in self.source_datapipe:
            if prev_filename is None:
                prev_filename = filename
            if line and prev_filename == filename:
                self.buffer.append(line)
            else:
                if self.buffer:
                    yield prev_filename, self.joiner(self.buffer)  # type: ignore[misc]
                if line:
                    self.buffer = [line]
                else:
                    self.buffer = []
                prev_filename = filename
        if self.buffer:
            yield prev_filename, self.joiner(self.buffer)  # type: ignore[misc]

    @final
    def reset(self) -> None:
        self.buffer = []

    def __getstate__(self):
        state = (self.source_datapipe, self.joiner)
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (self.source_datapipe, self.joiner) = state
        self.buffer = []

    def __del__(self):
        self.buffer.clear()
