# TorchArrow 

Torcharrow is a [Panda](https://github.com/pandas-dev/pandas) inspired DataFrame library in Python built on the [Apache Arrow](https://github.com/apache/arrow) columnar memory format and 
leveraging the [Velox vectorized engine](https://github.com/facebookexternal/f4d/) for loading, filtering, mapping, joining, aggregating, and 
otherwise manipulating tabular data on CPUs.

TorchArrow supports [PyTorch](https://github.com/pytorch/pytorch)'s Tensors as first class citizens. It allows mostly zero copy interop with Numpy, Pandas, PyArrow, CuDf and of coarse intgerates well with PyTorch datawrangling workflows.



## Install torcharrow
```
pip3 install .
```
## Test torcharrow
```
python3 -m unittest
```
## Documentation
All documentation is available via Notebooks:
* [TorchArrow in 10 minutes (a tutorial)](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow10min.ipynb)
* [TorchArrow mutability] (https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_mutability.ipynb)
* TorchArrow tracing and rerunning (TBD) 

## Status
This directory supports rapid development. So expect frequent changes.

Still to be done:
* Arrow and Panda interoperability
* Relational operators and fluent style programming
* Columns implemented by Velox and not in Python proper
* Add tabular.py as package in setup and not as code
* Implement tracing and rerunning

