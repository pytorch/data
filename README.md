# TorchData (see note below on current status)

[**Why TorchData?**](#why-composable-data-loading) | [**Install guide**](#installation) |
[**What are DataPipes?**](#what-are-datapipes) | [**Beta Usage and Feedback**](#beta-usage-and-feedback) |
[**Contributing**](#contributing) | [**Future Plans**](#future-plans)

**:warning: As of July 2023, we have paused active development on TorchData and have paused new releases. We have learnt
a lot from building it and hearing from users, but also believe we need to re-evaluate the technical design and approach
given how much the industry has changed since we began the project. During the rest of 2023 we will be re-evaluating our
plans in this space. Please reach out if you suggestions or comments (please use
[#1196](https://github.com/pytorch/data/issues/1196) for feedback).**

`torchdata` is a library of common modular data loading primitives for easily constructing flexible and performant data
pipelines.

This library introduces composable Iterable-style and Map-style building blocks called
[`DataPipes`](#what-are-datapipes) that work well out of the box with the PyTorch's `DataLoader`. These built-in
`DataPipes` have the necessary functionalities to reproduce many different datasets in TorchVision and TorchText, namely
loading files (from local or cloud), parsing, caching, transforming, filtering, and many more utilities. To understand
the basic structure of `DataPipes`, please see [What are DataPipes?](#what-are-datapipes) below, and to see how
`DataPipes` can be practically composed together into datasets, please see our
[examples](https://pytorch.org/data/main/examples.html).

On top of `DataPipes`, this library provides a new `DataLoader2` that allows the execution of these data pipelines in
various settings and execution backends (`ReadingService`). You can learn more about the new version of `DataLoader2` in
our [full DataLoader2 documentation](https://pytorch.org/data/main/dataloader2.html#dataloader2). Additional features
are work in progres, such as checkpointing and advanced control of randomness and determinism.

Note that because many features of the original DataLoader have been modularized into DataPipes, their source codes live
as [standard DataPipes in pytorch/pytorch](https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes)
rather than torchdata to preserve backward-compatibility support and functional parity within `torch`. Regardless, you
can to them by importing them from `torchdata`.

## Why composable data loading?

Over many years of feedback and organic community usage of the PyTorch `DataLoader` and `Dataset`, we've found that:

1. The original `DataLoader` bundled too many features together, making them difficult to extend, manipulate, or
   replace. This has created a proliferation of use-case specific `DataLoader` variants in the community rather than an
   ecosystem of interoperable elements.
2. Many libraries, including each of the PyTorch domain libraries, have rewritten the same data loading utilities over
   and over again. We can save OSS maintainers time and effort rewriting, debugging, and maintaining these commonly used
   elements.

These reasons inspired the creation of `DataPipe` and `DataLoader2`, with a goal to make data loading components more
flexible and reusable.

## Installation

### Version Compatibility

The following is the corresponding `torchdata` versions and supported Python versions.

| `torch`              | `torchdata`        | `python`          |
| -------------------- | ------------------ | ----------------- |
| `master` / `nightly` | `main` / `nightly` | `>=3.8`, `<=3.11` |
| `2.0.0`              | `0.6.0`            | `>=3.8`, `<=3.11` |
| `1.13.1`             | `0.5.1`            | `>=3.7`, `<=3.10` |
| `1.12.1`             | `0.4.1`            | `>=3.7`, `<=3.10` |
| `1.12.0`             | `0.4.0`            | `>=3.7`, `<=3.10` |
| `1.11.0`             | `0.3.0`            | `>=3.7`, `<=3.10` |

### Colab

Follow the instructions
[in this Colab notebook](https://colab.research.google.com/drive/1eSvp-eUDYPj0Sd0X_Mv9s9VkE8RNDg1u). The notebook also
contains a simple usage example.

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

You can then proceed to run [our examples](https://github.com/pytorch/data/tree/main/examples), such as
[the IMDb one](https://github.com/pytorch/data/blob/main/examples/text/imdb.py).

### From source

```bash
pip install .
```

If you'd like to include the S3 IO datapipes and aws-sdk-cpp, you may also follow
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
built-in features are implemented as `IterDataPipes`, we encourage the usage of built-in `IterDataPipe` as much as
possible and convert them to `MapDataPipe` only when necessary.

## DataLoader2

A new, light-weight DataLoader2 is introduced to decouple the overloaded data-manipulation functionalities from
`torch.utils.data.DataLoader` to `DataPipe` operations. Besides, certain features can only be achieved with
`DataLoader2`, such as like checkpointing/snapshotting and switching backend services to perform high-performant
operations.

Please read the [full documentation here](https://pytorch.org/data/main/dataloader2.html).

## Tutorial

A tutorial of this library is
[available here on the documentation site](https://pytorch.org/data/main/dp_tutorial.html). It covers four topics:
[using DataPipes](https://pytorch.org/data/main/dp_tutorial.html#using-datapipes),
[working with DataLoader](https://pytorch.org/data/main/dp_tutorial.html#working-with-dataloader),
[implementing DataPipes](https://pytorch.org/data/main/dp_tutorial.html#implementing-a-custom-datapipe), and
[working with Cloud Storage Providers](https://pytorch.org/data/main/dp_tutorial.html#working-with-cloud-storage-providers).

There is also a tutorial available on
[how to work with the new DataLoader2](https://pytorch.org/data/main/dlv2_tutorial.html).

## Usage Examples

We provide a simple usage example in this
[Colab notebook](https://colab.research.google.com/drive/1eSvp-eUDYPj0Sd0X_Mv9s9VkE8RNDg1u). It can also be downloaded
and executed locally as a Jupyter notebook.

In addition, there are several data loading implementations of popular datasets across different research domains that
use `DataPipes`. You can find a few [selected examples here](https://pytorch.org/data/main/examples.html).

## Frequently Asked Questions (FAQ)

<details>
<summary>
What should I do if the existing set of DataPipes does not do what I need?
</summary>

You can
[implement your own custom DataPipe](https://pytorch.org/data/main/dp_tutorial.html#implementing-a-custom-datapipe). If
you believe your use case is common enough such that the community can benefit from having your custom DataPipe added to
this library, feel free to open a GitHub issue. We will be happy to discuss!

</details>

<details>
<summary>
What happens when the <code>Shuffler</code> DataPipe is used with DataLoader?
</summary>

In order to enable shuffling, you need to add a `Shuffler` to your DataPipe line. Then, by default, shuffling will
happen at the point where you specified as long as you do not set `shuffle=False` within DataLoader.

</details>

<details>
<summary>
What happens when the <code>Batcher</code> DataPipe is used with DataLoader?
</summary>

If you choose to use `Batcher` while setting `batch_size > 1` for DataLoader, your samples will be batched more than
once. You should choose one or the other.

</details>

<details>
<summary>
Why are there fewer built-in <code>MapDataPipes</code> than <code>IterDataPipes</code>?
</summary>

By design, there are fewer `MapDataPipes` than `IterDataPipes` to avoid duplicate implementations of the same
functionalities as `MapDataPipe`. We encourage users to use the built-in `IterDataPipe` for various functionalities, and
convert it to `MapDataPipe` as needed.

</details>

<details>
<summary>
How is multiprocessing handled with DataPipes?
</summary>

Multi-process data loading is still handled by the `DataLoader`, see the
[DataLoader documentation for more details](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading).
As of PyTorch version >= 1.12.0 (TorchData version >= 0.4.0), data sharding is automatically done for DataPipes within
the `DataLoader` as long as a `ShardingFilter` DataPipe exists in your pipeline. Please see the
[tutorial](https://pytorch.org/data/main/dp_tutorial.html#working-with-dataloader) for an example.

</details>

<details>
<summary>
What is the upcoming plan for DataLoader?
</summary>

`DataLoader2` is in the prototype phase and more features are actively being developed. Please see the
[README file in `torchdata/dataloader2`](https://github.com/pytorch/data/blob/main/torchdata/dataloader2/README.md). If
you would like to experiment with it (or other prototype features), we encourage you to install the nightly version of
this library.

</details>

<details>
<summary>
Why is there an Error saying the specified DLL could not be found at the time of importing <code>portalocker</code>?
</summary>

It only happens for people who runs `torchdata` on Windows OS as a common problem with `pywin32`. And, you can find the
reason and the solution for it in the
[link](https://github.com/mhammond/pywin32#the-specified-procedure-could-not-be-found--entry-point-not-found-errors).

</details>

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## Beta Usage and Feedback

We'd love to hear from and work with early adopters to shape our designs. Please reach out by raising an issue if you're
interested in using this tooling for your project.

## License

TorchData is BSD licensed, as found in the [LICENSE](LICENSE) file.
