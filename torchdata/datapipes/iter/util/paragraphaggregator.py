# Copyright (c) Facebook, Inc. and its affiliates.
from torch.utils.data import IterDataPipe, functional_datapipe


def _default_line_join(lines):
    return "\n".join(lines)


@functional_datapipe("lines_to_paragraphs")
class ParagraphAggregatorIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, joiner=_default_line_join):
        self.source_datapipe = source_datapipe
        self.joiner = joiner

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
                    yield (prev_filename, self.joiner(buffer))
                if line:
                    buffer = [line]
                else:
                    buffer = []
                prev_filename = filename
        if buffer:
            yield (prev_filename, self.joiner(buffer))
