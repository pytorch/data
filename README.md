# TorchData

[**What is TorchData?**](#what-is-torchdata) | [**Stateful DataLoader**](#stateful-dataloader) |
[**Install guide**](#installation) | [**Contributing**](#contributing) | [**License**](#license)

##

## What is TorchData?

The TorchData project is an iterative enhancement to the PyTorch torch.utils.data.DataLoader and
torch.utils.data.Dataset/IterableDataset to make them scalable, performant dataloading solutions. We will be iterating
on the enhancements under [the torchdata repo](torchdata).

Our first change begins with adding checkpointing to torch.utils.data.DataLoader, which can be found in
[stateful_dataloader, a drop-in replacement for torch.utils.data.DataLoader](torchdata/stateful_dataloader), by defining
`load_state_dict` and `state_dict` methods that enable mid-epoch checkpointing, and an API for users to track custom
iteration progress, and other custom states from the dataloader workers such as token buffers and/or RNG states.

## Stateful DataLoader

`torchdata.stateful_dataloader.StatefulDataLoader` is a drop-in replacement for torch.utils.data.DataLoader which
provides state_dict and load_state_dict functionality. See
[the Stateful DataLoader main page](torchdata/stateful_dataloader) for more information and examples. Also check out the
examples
[in this Colab notebook](https://colab.research.google.com/drive/1tonoovEd7Tsi8EW8ZHXf0v3yHJGwZP8M?usp=sharing).

## torchdata.nodes

torchdata.nodes is a library of composable iterators (not iterables!) that let you chain together common dataloading and
pre-proc operations. It follows a streaming programming model, although "sampler + Map-style" can still be configured if
you desire. See [torchdata.nodes main page](torchdata/nodes) for more details. Stay tuned for tutorial on
torchdata.nodes coming soon!

## Installation

### Version Compatibility

The following is the corresponding `torchdata` versions and supported Python versions.

| `torch`              | `torchdata`        | `python`          |
| -------------------- | ------------------ | ----------------- |
| `master` / `nightly` | `main` / `nightly` | `>=3.9`, `<=3.13` |
| `2.6.0`              | `0.11.0`           | `>=3.9`, `<=3.13` |
| `2.5.0`              | `0.10.0`           | `>=3.9`, `<=3.12` |
| `2.5.0`              | `0.9.0`            | `>=3.9`, `<=3.12` |
| `2.4.0`              | `0.8.0`            | `>=3.8`, `<=3.12` |
| `2.0.0`              | `0.6.0`            | `>=3.8`, `<=3.11` |
| `1.13.1`             | `0.5.1`            | `>=3.7`, `<=3.10` |
| `1.12.1`             | `0.4.1`            | `>=3.7`, `<=3.10` |
| `1.12.0`             | `0.4.0`            | `>=3.7`, `<=3.10` |
| `1.11.0`             | `0.3.0`            | `>=3.7`, `<=3.10` |

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

### From source

```bash
pip install .
```

In case building TorchData from source fails, install the nightly version of PyTorch following the linked guide on the
[contributing page](CONTRIBUTING.md#install-pytorch-nightly).

### From nightly

The nightly version of TorchData is also provided and updated daily from main branch.

Using pip:

```bash
pip install --pre torchdata --index-url https://download.pytorch.org/whl/nightly/cpu
```

Using conda:

```bash
conda install torchdata -c pytorch-nightly
```

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## Beta Usage and Feedback

We'd love to hear from and work with early adopters to shape our designs. Please reach out by raising an issue if you're
interested in using this tooling for your project.

## License

TorchData is BSD licensed, as found in the [LICENSE](LICENSE) file.
