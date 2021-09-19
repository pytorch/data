# TorchData (🚨 Warning: Unstable Prototype 🚨)

[**Why torchdata?**](#why-composable-data-loading)
| [**Prototype Usage and Feedback**](#prototype-usage-and-feedback)
| [**Install guide**](#installation)
| [**Contributing**](#contributing)
| [**Future Plans**](#future-plans)

**This is prototype library currently under heavy development. It does not currently have stable releases, and as such will likely be modified significantly in BC-breaking ways until beta release (targeting early 2022), and can only be used with the PyTorch nighly binaries. If you have suggestions on the API or use cases you'd like to be covered, please open an github issue. We'd love to hear thoughts and feedback.**

`torchdata` is a prototype library of common modular data loading primitives for easily constructing flexible and performant data pipelines. 

It aims to provide composable iter-style and map-style [`DataPipes`](https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes) that work well out of the box with the PyTorch `DataLoader`. Right now it only contains basic functionality to reproduce several datasets in TorchVision and TorchText, namely including loading, parsing, caching, and several other utilities (e.g. hash checking). We plan to expand and harden this set considerably over the coming months. To understand how `DataPipes` can be composed into datasets, please see our [`examples/`](examples/) directory.

Note that many features of the original DataLoader have been modularized into DataPipes, and now live as [standard DataPipes in pytorch/pytorch](https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes) rather than torchdata to preserve BC functional parity within torch.

## Why composable data loading?

Over many years of feedback and organic community usage of the PyTorch DataLoader and DataSets, we've found that:

1. The original DataLoader bundled too many features together, making them difficult to extend, manipulate, or replace. This has created a proliferation of use-case specific DataLoader variants in the community rather than an ecosystem of interoperable elements.
2. Many libraries, including each of the PyTorch domain libraries, have rewritten the same data loading utilities over and over again. We can save OSS maintainers time and effort rewriting, debugging, and maintaining these table-stakes elements.

## Prototype Usage and Feedback

To understand how `DataPipes` can be composed into datasets, please see our [`examples/`](examples/) directory.

We'd love to hear from and work with early adopters to shape the design. Please reach out on the issue tracker if you're interested in using this for your project.

## Installation

### Colab

Follow the instructions [in this Colab notebook (TODO)](https://colab.research.google.com/drive/1x1ESG0_N02txFuQwyTfCnjhqzS-PzQjA#scrollTo=SVnu66W-wQfF)

### Local pip or conda (TODO: Finish)

First, set up an environment. We will be installing a nightly PyTorch binary
as well as torchdata. If you're using conda, create a conda environment:
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
# For CUDA 11.1
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# For CPU-only build
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```
If you already have a nightly of PyTorch installed and wanted to upgrade it
(recommended!), append `--upgrade` to one of those commands.

Install torchdata:
```
pip install --user "git+https://github.com/pytorch/data.git"
```

Run a quick sanity check in python:
```py
>>> from torchdata.datapipes.iter import HttpReader
>>> URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
>>> ag_news_train = HttpReader([URL]).parse_csv_files().map(lambda t: (int(t[1]), " ".join(t[2:])))
>>> agn_batches = ag_news_train.batch(2).map(lambda batch: {'labels': [sample[0] for sample in batch],\
                                      'text': [sample[1].split() for sample in batch]})
>>> first_batch = next(iter(agn_batches))
>>> assert batch['text'][0][0:8] == ['Wall', 'St.', 'Bears', 'Claw', 'Back', 'Into', 'the', 'Black']
```

### From source

```bash
$ pip install -e git+https://github.com/pytorch/torchdata
```

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## Future Plans

We hope to sufficiently expand the library, harden APIs, and gather feedback to enable a beta release at the time of the PyTorch 1.11 release (early 2022).

## License

TorchData is BSD licensed, as found in the [LICENSE](LICENSE) file.
