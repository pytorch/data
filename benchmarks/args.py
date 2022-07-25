from dataclasses import dataclass, fields
from enum import Enum

from simple_parsing import ArgumentParser


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "gtsrb"  # TODO: Integrate with HF datasets
    model_name: str = "resnext50_32x4d"  # TODO: torchvision models supported only
    batch_size: int = 1
    device: str = "cuda:0"  # Options are cpu or cuda:0
    num_epochs: int = 1
    report_location: str = "report.csv"
    num_wokers: int = 1
    shuffle: bool = True
    dataloader_version: int = 1  # Options are 1 or 2


## Arg parsing
def arg_parser():
    parser = ArgumentParser()
    parser.add_arguments(BenchmarkConfig, dest="options")
    args = parser.parse_args()
    benchmark_config = args.options
    return benchmark_config


if __name__ == "__main__":
    arg_parser()
