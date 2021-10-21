# Copyright (c) Facebook, Inc. and its affiliates.
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from typing import Callable, List, Tuple, TypeVar


T_co = TypeVar("T_co", covariant=True)


def _default_line_join(lines: List[str]) -> str:
    return "\n".join(lines)


@functional_datapipe("lines_to_paragraphs")
class ParagraphAggregatorIterDataPipe(IterDataPipe[T_co]):
    r"""
    Iterable DataPipe that aggregates lines of text from the same file into a single paragraph.
    Specifically, this accepts a DataPipe consisting of tuples of a file name and a line. For each tuple,
    it checks if the file name matches the file name from the previous tuple. If yes, it joins the current line
    with existing paragraph. If the file names do not match, the existing paragraph is yielded and a new
    paragraph starts.

    Args:
        source_datapipe: a DataPipe with tuples of a file name and a line
        joiner: a function that joins a list of lines together
    """
    def __init__(self, source_datapipe: IterDataPipe[Tuple[str, T_co]], joiner: Callable = _default_line_join) -> None:
        self.source_datapipe: IterDataPipe[Tuple[str, T_co]] = source_datapipe
        self.joiner: Callable = joiner

    def __iter__(self):
        buffer = []
        prev_filename = None
        for filename, line in self.source_datapipe:
            if prev_filename is None:
                prev_filename = filename
            if line and prev_filename == filename:
                buffer.append(line)
            else:
                if buffer:
                    yield prev_filename, self.joiner(buffer)
                if line:
                    buffer = [line]
                else:
                    buffer = []
                prev_filename = filename
        if buffer:
            yield prev_filename, self.joiner(buffer)
