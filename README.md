# TorchData (🚨 Warning: Unstable Prototype 🚨)

[**Why torchdata?**](#why-composable-data-loading) | [**Install guide**](#installation) |
[**What are DataPipes?**](#what-are-datapipes) | [**Prototype Usage and Feedback**](#prototype-usage-and-feedback) |
[**Contributing**](#contributing) | [**Future Plans**](#future-plans)

**This is a prototype library currently under heavy development. It does not currently have stable releases, and as such
will likely be modified significantly in BC-breaking ways until beta release (targeting early 2022), and can only be
used with the PyTorch nightly binaries. If you have suggestions on the API or use cases you'd like to be covered, please
open a github issue. We'd love to hear thoughts and feedback.**

`torchdata` is a prototype library of common modular data loading primitives for easily constructing flexible and
performant data pipelines.

It aims to provide composable iter-style and map-style building blocks called [`DataPipes`](#what-are-datapipes) that
work well out of the box with the PyTorch `DataLoader`. Right now it only contains basic functionality to reproduce
several datasets in TorchVision and TorchText, namely including loading, parsing, caching, and several other utilities
(e.g. hash checking). We plan to expand and harden this set considerably over the coming months.

To understand the basic structure of `DataPipes`, please see [What are DataPipes?](#what-are-datapipes) below, and to
see how `DataPipes` can be practically composed into datasets, please see our [`examples/`](examples/) directory.

Note that because many features of the original DataLoader have been modularized into DataPipes, some now live as
[standard DataPipes in pytorch/pytorch](https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes)
rather than torchdata to preserve BC functional parity within torch.

## Why composable data loading?

Over many years of feedback and organic community usage of the PyTorch DataLoader and DataSets, we've found that:

1. The original DataLoader bundled too many features together, making them difficult to extend, manipulate, or replace.
   This has created a proliferation of use-case specific DataLoader variants in the community rather than an ecosystem
   of interoperable elements.
2. Many libraries, including each of the PyTorch domain libraries, have rewritten the same data loading utilities over
   and over again. We can save OSS maintainers time and effort rewriting, debugging, and maintaining these table-stakes
   elements.

## Installation

### Colab

Follow the instructions
[in this Colab notebook](https://colab.research.google.com/drive/1x1ESG0_N02txFuQwyTfCnjhqzS-PzQjA)

### Local pip or conda

First, set up an environment. We will be installing a nightly PyTorch binary as well as torchdata. If you're using
conda, create a conda environment:

```
conda create --name torchdata
conda activate torchdata
```

If you wish to use `venv` instead:

```
python -m venv torchdata-env
source torchdata-env/bin/activate
```

Next, install one of the following following PyTorch nightly binaries.

```
# For CUDA 10.2
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
# For CUDA 11.3
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
# For CPU-only build
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

If you already have a nightly of PyTorch installed and wanted to upgrade it (recommended!), append `--upgrade` to one of
those commands.

Install torchdata:

```
pip install --user "git+https://github.com/pytorch/data.git"
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
$ pip install -e git+https://github.com/pytorch/data#egg=torchdata
```

## Building the Documentation

To build the documentation, you will need [Sphinx](http://www.sphinx-doc.org) and the PyTorch theme.

```bash
cd docs/
pip install -r requirements.txt
```

You can then build the documentation by running `make <format>` from the `docs/` folder. Run `make` to get a list of all
available output formats.

```bash
make html
```

## What are DataPipes?

Early on, we observed widespread confusion between the PyTorch `DataSets` which represented reusable loading tooling
(e.g. TorchVision's [`ImageFolder`](https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L272)),
and those that represented pre-built iterators/accessors over actual data corpora (e.g. TorchVision's
[ImageNet](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py#L20)). This led to an
unfortunate pattern of siloed inheritence of data tooling rather than composition.

`DataPipe` is simply a renaming and repurposing of the PyTorch `DataSet` for composed usage. A `DataPipe` takes in some
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
            yield file_name, json.loads(data)

    def __len__(self):
        return len(self.source_datapipe)
```

You can see in this example how DataPipes can be easily chained together to compose graphs of transformations that
reproduce sohpisticated data pipelines, with streamed operation as a first-class citizen.

Under this naming convention, `DataSet` simply refers to a graph of `DataPipes`, and a dataset module like `ImageNet`
can be rebuilt as a factory function returning the requisite composed `DataPipes`. Note that the vast majority of
initial support is focused on `IterDataPipes`, while more `MapDataPipes` support will come later.

### Implementing DataPipes

As a guiding example, let's implement an `IterDataPipe` that applies a callable to the input iterator. For
`MapDataPipe`s, take a look at the [map](https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes/map)
folder for examples, and follow the steps below for the `__getitem__` method instead of `__iter__`.

#### Naming

The naming convention for `DataPipe`s is "Operation"-er, followed by `IterDataPipe` or `MapDataPipe`, as each DataPipe
is essentially a container to apply an operation to data yielded from a source DataPipe. For succinctness, we alias to
just "Operation-er" in **init** files. For our `IterDataPipe` example, we'll name the module `MapperIterDataPipe` and
alias it as `iter.Mapper` under `datapipes`.

#### Constructor

DataSets are now generally constructed as stacks of `DataPipes`, so each `DataPipe` typically takes a source `DataPipe`
as its first argument.

```py
class MapperIterDataPipe(IterDataPipe):
    def __init__(self, dp, fn) -> None:
        super().__init__()
        self.dp = dp
        self.fn = fn
```

Note:

- Avoid loading data from the source DataPipe in `__init__` function, in order to support lazy data loading and save
  memory.
- If `IterDataPipe` instance holds data in memory, please be ware of the in-place modification of data. When second
  iterator is created from the instance, the data may have already changed. Please take
  [`IterableWrapper`](https://github.com/pytorch/pytorch/blob/master/torch/utils/data/datapipes/iter/utils.py) class as
  reference to `deepcopy` data for each iterator.

#### Iterator

For `IterDataPipes`, an `__iter__` function is needed to consume data from the source `IterDataPipe` then apply the
operation over the data before `yield`.

```py
class MapperIterDataPipe(IterDataPipe):
    ...

    def __iter__(self):
        for d in self.dp:
            yield self.fn(d)
```

#### Length

In many cases, as in our `MapperIterDataPipe` example, the `__len__` method of a DataPipe returns the length of the
source DataPipe.

```py
class MapperIterDataPipe(IterDataPipe):
    ...

    def __len__(self):
        return len(self.dp)
```

However, note that `__len__` is optional for `IterDataPipe` and often inadvisable. For `CSVParserIterDataPipe` in the
[using DataPipes section below](#using-datapipes), `__len__` is not implemented because the number of rows in each file
is unknown before loading it. In some special cases, `__len__` can be made to either return an integer or raise an Error
depending on the input. In those cases, the Error must be a `TypeError` to support Python's build-in functions like
`list(dp)`.

#### Registering DataPipes with the functional API

Each DataPipe can be registered to support functional invocation using the decorator `functional_datapipe`.

```py
@functional_datapipe("map")
class MapperIterDataPipe(IterDataPipe):
    ...
```

The stack of DataPipes can then be constructed in functional form:

```py
>>> import torch.utils.data.datapipes as dp
>>> datapipes1 = dp.iter.FileOpener(['a.file', 'b.file']).map(fn=decoder).shuffle().batch(2)

>>> datapipes2 = dp.iter.FileOpener(['a.file', 'b.file'])
>>> datapipes2 = dp.iter.Mapper(datapipes2)
>>> datapipes2 = dp.iter.Shuffler(datapipes2)
>>> datapipes2 = dp.iter.Batcher(datapipes2, 2)
```

In the above example, `datapipes1` and `datapipes2` represent the exact same stack of `IterDataPipe`s.

### Using DataPipes

For a complete example, suppose we want to load data from CSV files with the following steps:

- List all csv files in a directory
- Load csv files
- Parse csv file and yield rows

To support the above pipeline, `CSVParser` is registered as `parse_csv` to consume file streams and expand them as rows.

```py
@functional_datapipe("parse_csv")
class CSVParserIterDataPipe(IterDataPipe):
    def __init__(self, dp, **fmtparams) -> None:
        self.dp = dp
        self.fmtparams = fmtparams

    def __iter__(self):
        for filename, stream in self.dp:
            reader = csv.reader(stream, **self.fmtparams)
            for row in reader:
                yield filename, row
```

Then, the pipeline can be assembled as follows:

```py
>>> import torch.utils.data.datapipes as dp

>>> FOLDER = 'path/2/csv/folder'
>>> datapipe = dp.iter.FileLister([FOLDER]).filter(fn=lambda filename: filename.endswith('.csv'))
>>> datapipe = dp.iter.FileOpener(datapipe, mode='rt')
>>> datapipe = datapipe.parse_csv(delimiter=',')

>>> for d in datapipe: # Start loading data
...     pass
```

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## Prototype Usage and Feedback

We'd love to hear from and work with early adopters to shape our designs. Please reach out by raising an issue if you're
interested in using this tooling for your project.

## Future Plans

We hope to sufficiently expand the library, harden APIs, and gather feedback to enable a beta release at the time of the
PyTorch 1.11 release (early 2022).

## License

TorchData is BSD licensed, as found in the [LICENSE](LICENSE) file.
