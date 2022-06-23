# TorchData

[**Why torchdata?**](#why-composable-data-loading) | [**Install guide**](#installation) |
[**What are DataPipes?**](#what-are-datapipes) | [**Beta Usage and Feedback**](#beta-usage-and-feedback) |
[**Contributing**](#contributing) | [**Future Plans**](#future-plans)

**This library is currently in the Beta stage and currently does not have a stable release. The API may change based on
user feedback or performance. We are committed to bring this library to stable release, but future changes may not be
completely backward compatible. If you install from source or use the nightly version of this library, use it along with
the PyTorch nightly binaries. If you have suggestions on the API or use cases you'd like to be covered, please open a
GitHub issue. We'd love to hear thoughts and feedback.**

`torchdata` is a library of common modular data loading primitives for easily constructing flexible and performant data
pipelines.

It aims to provide composable Iterable-style and Map-style building blocks called [`DataPipes`](#what-are-datapipes)
that work well out of the box with the PyTorch's `DataLoader`. It contains functionality to reproduce many different
datasets in TorchVision and TorchText, namely including loading, parsing, caching, and several other utilities (e.g.
hash checking). We will continue to expand and harden this set of API based on user feedback.

To understand the basic structure of `DataPipes`, please see [What are DataPipes?](#what-are-datapipes) below, and to
see how `DataPipes` can be practically composed into datasets, please see our [`examples/`](examples/) directory.

Note that because many features of the original DataLoader have been modularized into DataPipes, some now live as
[standard DataPipes in pytorch/pytorch](https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes)
rather than torchdata to preserve BC functional parity within torch.

## Why composable data loading?

Over many years of feedback and organic community usage of the PyTorch `DataLoader` and `Dataset`, we've found that:

1. The original `DataLoader` bundled too many features together, making them difficult to extend, manipulate, or
   replace. This has created a proliferation of use-case specific `DataLoader` variants in the community rather than an
   ecosystem of interoperable elements.
2. Many libraries, including each of the PyTorch domain libraries, have rewritten the same data loading utilities over
   and over again. We can save OSS maintainers time and effort rewriting, debugging, and maintaining these table-stakes
   elements.

## Installation

### Version Compatibility

The following is the corresponding `torchdata` versions and supported Python versions.

| `torch`            | `torchdata`        | `python`          |
| ------------------ | ------------------ | ----------------- |
| `main` / `nightly` | `main` / `nightly` | `>=3.7`, `<=3.10` |
| `1.11.0`           | `0.3.0`            | `>=3.7`, `<=3.10` |

### Colab

Follow the instructions
[in this Colab notebook](https://colab.research.google.com/drive/1x1ESG0_N02txFuQwyTfCnjhqzS-PzQjA)

### Local pip or conda

First, set up an environment. We will be installing a PyTorch binary as well as torchdata. If you're using conda, create
a conda environment:

```bash
conda create --name torchdata
conda activate torchdata
```

If you wish to use `venv` instead:

```bash
python -m venv torchdata-env
source torchdata-env/bin/activate
```

Install torchdata:

Using pip:

```bash
pip install torchdata
```

Using conda:

```bash
conda install -c pytorch torchdata
```

Run a quick sanity check in python:

```py
from torchdata.datapipes.iter import HttpReader
URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
ag_news_train = HttpReader([URL]).parse_csv().map(lambda t: (int(t[0]), " ".join(t[1:])))
agn_batches = ag_news_train.batch(2).map(lambda batch: {'labels': [sample[0] for sample in batch],\
                                      'text': [sample[1].split() for sample in batch]})
batch = next(iter(agn_batches))
assert batch['text'][0][0:8] == ['Wall', 'St.', 'Bears', 'Claw', 'Back', 'Into', 'the', 'Black']
```

### From source

```bash
python setup.py install
```

In you'd like to include the S3 IO datapipes and aws-sdk-cpp, you may also follow
[the instructions here](https://github.com/pytorch/data/blob/main/torchdata/datapipes/iter/load/README.md)

In case building TorchData from source fails, install the nightly version of PyTorch following the linked guide on the
[contributing page](https://github.com/pytorch/data/blob/main/CONTRIBUTING.md#install-pytorch-nightly).

### From nightly

The nightly version of TorchData is also provided and updated daily from main branch.

Using pip:

```bash
pip install --pre torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Using conda:

```bash
conda install torchdata -c pytorch-nightly
```

## What are DataPipes?

Early on, we observed widespread confusion between the PyTorch `Dataset` which represented reusable loading tooling
(e.g. [TorchVision's `ImageFolder`](https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L272)),
and those that represented pre-built iterators/accessors over actual data corpora (e.g. TorchVision's
[ImageNet](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py#L21)). This led to an
unfortunate pattern of siloed inheritance of data tooling rather than composition.

`DataPipe` is simply a renaming and repurposing of the PyTorch `Dataset` for composed usage. A `DataPipe` takes in some
access function over Python data structures, `__iter__` for `IterDataPipes` and `__getitem__` for `MapDataPipes`, and
returns a new access function with a slight transformation applied. For example, take a look at this `JsonParser`, which
accepts an IterDataPipe over file names and raw streams, and produces a new iterator over the filenames and deserialized
data:

```py
import json

class JsonParserIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, **kwargs) -> None:
        self.source_datapipe = source_datapipe
        self.kwargs = kwargs

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            data = stream.read()
            yield file_name, json.loads(data, **self.kwargs)

    def __len__(self):
        return len(self.source_datapipe)
```

You can see in this example how DataPipes can be easily chained together to compose graphs of transformations that
reproduce sophisticated data pipelines, with streamed operation as a first-class citizen.

Under this naming convention, `Dataset` simply refers to a graph of `DataPipes`, and a dataset module like `ImageNet`
can be rebuilt as a factory function returning the requisite composed `DataPipes`. Note that the vast majority of
initial support is focused on `IterDataPipes`, while more `MapDataPipes` support will come later.

## Tutorial

A tutorial of this library is [available here on the documentation site](https://pytorch.org/data/main/tutorial.html).
It covers three topics: [using DataPipes](https://pytorch.org/data/main/tutorial.html#using-datapipes),
[working with DataLoader](https://pytorch.org/data/main/tutorial.html#working-with-dataloader), and
[implementing DataPipes](https://pytorch.org/data/main/tutorial.html#implementing-a-custom-datapipe).

## Usage Examples

There are several data loading implementations of popular datasets across different research domains that use
`DataPipes`. You can find a few [selected examples here](https://pytorch.org/data/main/examples.html).

## Frequently Asked Questions (FAQ)

Q: What should I do if the existing set of DataPipes does not do what I need?

A: You can
[implement your own custom DataPipe](https://pytorch.org/data/main/tutorial.html#implementing-a-custom-datapipe). If you
believe your use case is common enough such that the community can benefit from having your custom DataPipe added to
this library, feel free to open a GitHub issue. We will be happy to discuss!

Q: What happens when the `Shuffler` DataPipe is used with DataLoader?

A. In order to enable shuffling, you need to add a `Shuffler` to your DataPipe line. Then, by default, shuffling will
happen at the point where you specified as long as you do not set `shuffle=False` within DataLoader.

Q: What happens when the `Batcher` DataPipe is used with DataLoader?

A: If you choose to use `Batcher` while setting `batch_size > 1` for DataLoader, your samples will be batched more than
once. You should choose one or the other.

Q: Why are there fewer built-in `MapDataPipes` than `IterDataPipes`?

A: By design, there are fewer `MapDataPipes` than `IterDataPipes` to avoid duplicate implementations of the same
functionalities as `MapDataPipe`. We encourage users to use the built-in `IterDataPipe` for various functionalities, and
convert it to `MapDataPipe` as needed.

Q: How is multiprocessing handled with DataPipes?

A: Multi-process data loading is still handled by the `DataLoader`, see the
[DataLoader documentation for more details](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading).
As of PyTorch version >= 1.12.0 (TorchData version >= 0.4.0), data sharding is automatically done for DataPipes within
the `DataLoader` as long as a `ShardingFiler` DataPipe exists in your pipeline. Please see the
[tutorial](https://pytorch.org/data/beta/tutorial.html#working-with-dataloader) for an example.

Q: What is the upcoming plan for DataLoader?

A: DataLoader2 is in the prototype phase and more features are actively being developed. Please see the README file in
`torchdata/dataloader2`. If you would like to experiment with it (or other prototype features), we encourage you to
install the nightly version of this library.

Q: Why is there an Error saying the specified DLL could not be found at the time of importing `portalocker`?

A: It only happens for people who runs `torchdata` on Windows OS as a common problem with `pywin32`. And, you can find
the reason and the solution for it in the
[link](https://github.com/mhammond/pywin32#the-specified-procedure-could-not-be-found--entry-point-not-found-errors).

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## Beta Usage and Feedback

We'd love to hear from and work with early adopters to shape our designs. Please reach out by raising an issue if you're
interested in using this tooling for your project.

## Future Plans

We hope to continue to expand the library, harden APIs, and gather feedback to enable another release at the time of the
PyTorch 1.12 release (mid 2022). We also plan to release a new version of DataLoader by then. Stay tuned!

## License

TorchData is BSD licensed, as found in the [LICENSE](LICENSE) file.
