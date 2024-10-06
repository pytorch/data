# Contributing to TorchData

We want to make contributing to this project as easy and transparent as possible.

## TL;DR

We appreciate all contributions. If you are interested in contributing to TorchData, there are many ways to help out.
Your contributions may fall into the following categories:

- It helps the project if you can

  - Report issues that you're facing
  - Give a :+1: on issues that others reported and that are relevant to you

- Answering questions on the issue tracker, investigating bugs are very valuable contributions to the project.

- You would like to improve the documentation. This is no less important than improving the library itself! If you find
  a typo in the documentation, do not hesitate to submit a GitHub pull request.

- If you would like to fix a bug:

  - comment on the issue that you want to work on this issue
  - send a PR with your fix, see below.

- If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the
  feature with us.
  - We have a checklist of things to go through while adding a new DataPipe. See below.
- If you would like to feature a usage example in our documentation, discuss that with us in an issue.

## Issues

We use GitHub issues to track public bugs. Please follow the existing templates if possible and ensure that the
description is clear and has sufficient instructions to be able to reproduce the issue.

For question related to the usage of this library, please post a question on the
[PyTorch forum, under the "data" category](https://discuss.pytorch.org/c/data/37).

## Development installation

### Install PyTorch Nightly

```bash
conda install pytorch -c pytorch-nightly
# or with pip (see https://pytorch.org/get-started/locally/)
# pip install numpy
# pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```

### Install TorchData

```bash
git clone https://github.com/pytorch/data.git
cd data
pip install -e .
pip install flake8 typing mypy pytest expecttest
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation and examples.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

### Code style

`torchdata` enforces a fairly strict code format through [`pre-commit`](https://pre-commit.com). You can install it with

```shell
pip install pre-commit
```

or

```shell
conda install -c conda-forge pre-commit
```

To check and in most cases fix the code format, stage all your changes (`git add`) and run `pre-commit run`. To perform
the checks automatically before every `git commit`, you can install them with `pre-commit install`.

### Adding a new DataPipe

When adding a new DataPipe, there are few things that need to be done to ensure it is working and documented properly.

1. Naming - please follow the naming convention as
   [described here](https://pytorch.org/data/main/dp_tutorial.html#naming).
2. Testing - please add unit tests to ensure that the DataPipe is functioning properly. Here are the
   [test requirements](https://github.com/pytorch/data/issues/106) that we have.
   - One test that is commonly missed is the serialization test. Please add the new DataPipe to
     [`test_serialization.py`](https://github.com/pytorch/data/blob/main/test/test_serialization.py).
   - If your test requires interacting with files in the file system (e.g. opening a `csv` or `tar` file), we prefer
     those files to be generated during the test (see
     [`test_local_io.py`](https://github.com/pytorch/data/blob/main/test/test_local_io.py)). If the file is on a remote
     server, see [`test_remote_io.py`](https://github.com/pytorch/data/blob/main/test/test_remote_io.py).
3. Documentation - ensure that the DataPipe has docstring, usage example, and that it is added to the right category of
   the right RST file (in [`docs/source`](https://github.com/pytorch/data/tree/main/docs/source)) to be rendered.
   - If your DataPipe has a functional form (i.e. `@functional_datapipe(...)`), include it at the
     [end of the first sentence](https://github.com/pytorch/data/blob/main/torchdata/datapipes/iter/util/combining.py#L25)
     of your docstring. This will make sure it correctly shows up in the
     [summary table](https://pytorch.org/data/main/torchdata.datapipes.iter.html#archive-datapipes) of our
     documentation.
   - For usage examples we support both standard doctest-style interactive Python sessions and code-output-style blocks.
     See [sphinx doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) for more information. To
     build the documentation and validate that your example works please refer to
     [`docs`](https://github.com/pytorch/data/tree/main/docs).
4. Import - import the DataPipe in the correct `__init__.py` file and add it to the `__all__` list.
5. Interface - if the DataPipe has a functional form, make sure that is generated properly by `gen_pyi.py` into the
   relevant interface file.
   - You can re-generate the pyi files by re-running `pip install -e .`, then you can examine the new outputs.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of
Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to TorchData, you agree that your contributions will be licensed under the LICENSE file in the root
directory of this source tree.
