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
* [TorchArrow mutability](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_mutability.ipynb)
* [TorchArrow data pipes](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_data_pipes.ipynb)
* [TorchArrow tracing and analysis](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_traces.ipynb)
* TorchArrow multitargeting - TBD
* TorchArrow, Pandas, UPM and SQL: What's the difference - TBD
* TorchArrow, Design rationale - TBD

## Status
This directory supports rapid development. So expect frequent changes.

Still to be done:
* Add tabular.py as package in setup and not as code
* How to do Multitargetting
* An example program analysis (types/PPF?)
* Add example UDFs
* Add Tensors as example UDTs
* Using Numba for Jitting


