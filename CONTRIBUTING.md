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
  - We have a checklist of things to go through while adding a new Node. See below.
- If you would like to feature a usage example in our documentation, discuss that with us in an issue.

## Issues

We use GitHub issues to track public bugs. Please follow the existing templates if possible and ensure that the
description is clear and has sufficient instructions to be able to reproduce the issue.

For question related to the usage of this library, please post a question on the
[PyTorch forum, under the "data" category](https://discuss.pytorch.org/c/data/37).

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation and examples.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

## Development installation

### Install PyTorch Nightly

```bash
conda install pytorch -c pytorch-nightly
# or with pip (see https://pytorch.org/get-started/locally/)
# pip install numpy
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Install TorchData and Test Requirements

```bash
git clone https://github.com/pytorch/data.git
cd data
pip install -e .
pip install -r test/requirements.txt
```

### Code style

`torchdata` enforces a fairly strict code format through [`pre-commit`](https://pre-commit.com). You can install it with

```bash
conda install -c conda-forge pre-commit
# or pip install pre-commit
cd data
with-proxy conda install pre-commit
pre-commit install --install-hooks
```

### Running mypy and unit-tests locally

Currently we don't run mypy as part of pre-commit hooks

```bash
mypy --config-file=mypy.ini
```

```bash
pytest --durations=0 --no-header -v test/nodes/
```

### Adding a new Node

When adding a new Node, there are few things that need to be done to ensure it is working and documented properly.

The following simplifying assumptions are made of node implementations:

- state is managed solely by the BaseNode, not through any iterators returned from them.
- state_dict() returns the state of the most recently requested iterator.
- load_state_dict() will set the state for the next iterator.

1. Functionality - Nodes must subclass BaseNode and implement the required methods.

   - `.iterator(self, initial_state: Optional[Dict[str, Any]])` - return a new iterator/generator that is properly
     initialized with the optional initial_state
   - `.get_state(self) -> Dict[str, Any]` - return a dictionary representing the state of the most recently returned
     iterator, or if not yet requested, the initial state.
   - ensure you're calling `state_dict()/load_state_dict()` on ancestor BaseNodes. Here is a simple example of a pretty
     useless node:

   ```python
   class MyNode(BaseNode[T]):
     def __init__(self, parent: BaseNode[T]):
       self.parent = parent
       self.idx = 0  # not very useful state

     def iterator(self, initial_state: Optional[Dict[str, Any]]) -> Iterator[T]
       if initial_state is not None:
         self.parent.load_state_dict(initial_state["parent"])
         self.idx = initial_state["idx"]

       for item in self.parent:
         self.idx += 1
         yield item

     def get_state(self) -> Dict[str, Any]:
       return {
         "parent": self.parent.state_dict(),  # note we call state_dict() and not get_state() here
         "idx": self.idx,
       }
   ```

2. Typing - Include type-hints for all public functions and methods
3. Testing - please add unit tests to ensure that the Node is functioning properly.
   - In addition to testing basic functionatity, state management must also be tested.
   - For basic state testing, you may use `test.nodes.utils.run_test_save_load_state`. See `test/nodes/test_batch.py`
     for an example.
4. Documentation - ensure that the Node has a docstring, and a usage example.
5. Import - import the Node in the correct `__init__.py` file.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of
Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to TorchData, you agree that your contributions will be licensed under the LICENSE file in the root
directory of this source tree.
