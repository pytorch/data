import csv
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, TypeVar, Union

from torchdata.nodes.base_node import BaseNode, T


class CSVReader(BaseNode[Union[List[str], Dict[str, str]]]):
    """Node for reading CSV files with state management and header support.
    Args:
        file_path: Path to CSV file
        has_header: Whether first row contains column headers
        delimiter: CSV field delimiter
        return_dict: Return rows as dictionaries (requires has_header=True)
    """

    LINE_NUM_KEY = "line_num"
    HEADER_KEY = "header"

    def __init__(
        self,
        file_path: str,
        has_header: bool = False,
        delimiter: str = ",",
        return_dict: bool = False,
    ):
        super().__init__()
        self.file_path = file_path
        self.has_header = has_header
        self.delimiter = delimiter
        self.return_dict = return_dict
        if return_dict and not has_header:
            raise ValueError("return_dict=True requires has_header=True")
        self._file: Optional[TextIO] = None
        self._reader: Optional[Iterator[Union[List[str], Dict[str, str]]]] = None
        self._header: Optional[Sequence[str]] = None
        self._line_num: int = 0
        self.reset()  # Initialize reader

    def reset(self, initial_state: Optional[Dict[str, Any]] = None):
        super().reset(initial_state)

        if self._file and not self._file.closed:
            self._file.close()

        self._file = open(self.file_path, newline="", encoding="utf-8")
        self._line_num = 0

        if initial_state:
            self._header = initial_state.get(self.HEADER_KEY)
            target_line_num = initial_state[self.LINE_NUM_KEY]

            if self.return_dict:
                if self._header is None:
                    raise ValueError("return_dict=True requires has_header=True")
                self._reader = csv.DictReader(
                    self._file, delimiter=self.delimiter, fieldnames=self._header
                )
            else:
                self._reader = csv.reader(self._file, delimiter=self.delimiter)

            assert isinstance(self._reader, Iterator)
            if self.has_header:
                next(self._reader)  # Skip header
            for _ in range(target_line_num - self._line_num):
                try:
                    next(self._reader)
                    self._line_num += 1
                except StopIteration:
                    break
        else:

            if self.return_dict:
                self._reader = csv.DictReader(self._file, delimiter=self.delimiter)
                self._header = self._reader.fieldnames
            else:
                self._reader = csv.reader(self._file, delimiter=self.delimiter)
                if self.has_header:
                    self._header = next(self._reader)

    def next(self) -> Union[List[str], Dict[str, str]]:
        try:
            assert isinstance(self._reader, Iterator)
            row = next(self._reader)
            self._line_num += 1
            return row

        except StopIteration:
            self.close()
            raise

    def get_state(self) -> Dict[str, Any]:
        return {self.LINE_NUM_KEY: self._line_num, self.HEADER_KEY: self._header}

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
