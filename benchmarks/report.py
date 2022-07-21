import csv
from abc import ABC, abstractclassmethod
from dataclasses import dataclass, fields
from statistics import mean
from typing import Dict, list, tuple

import numpy as np

duration = int


@dataclass
class MetricCache:
    epoch_durations: list[duration]
    batch_durations: list[duration]
    total_duration: int = 0


class MetricExporter(ABC):
    @abstractclassmethod
    def export(self, metric_cache: MetricCache) -> None:
        return NotImplementedError

    def calculate_percentiles(self, metric_cache: MetricCache) -> Dict[str, float]:
        output = {}
        for field in fields(metric_cache):
            duration_list = getattr(metric_cache, field.name)
            percentiles = [
                np.percentile(duration_list, 0.5),
                np.percentile(duration_list, 0.9),
                np.percentile(duration_list, 0.99),
            ]
            output[field.name] = percentiles
        return output


class StdOutReport(MetricExporter):
    def export(self, metric_cache):
        percentiles_dict = metric_cache.calculate_percentiles()
        for field, percentiles in percentiles_dict.items:
            print(f"{field} duration is {percentiles}")


class CSVReport(MetricExporter):
    def export(self, metric_cache: MetricCache, filepath: str):
        percentiles_dict = metric_cache.calculate_percentiles()
        with open(filepath, "w") as file:
            writer = csv.writer(file)
            for field, percentiles in percentiles_dict.items:
                writer.writerow(field + percentiles)
