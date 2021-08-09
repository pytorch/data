import csv

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe('parse_csv_files')
class CSVParserIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            reader = csv.reader(stream)
            for row in reader:
                yield [file_name] + row
