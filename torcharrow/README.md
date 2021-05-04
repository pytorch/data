# TorchArrow 

Torcharrow is a [Panda](https://github.com/pandas-dev/pandas) inspired DataFrame library in Python built on the [Apache Arrow](https://github.com/apache/arrow) columnar memory format and 
leveraging the [Velox vectorized engine](https://github.com/facebookexternal/f4d/) for loading, filtering, mapping, joining, aggregating, and 
otherwise manipulating tabular data on CPUs.

TorchArrow supports [PyTorch](https://github.com/pytorch/pytorch)'s Tensors as first class citizens. It allows mostly zero copy interop with Numpy, Pandas, PyArrow, CuDf and of coarse intgerates well with PyTorch datawrangling workflows.



## Install torcharrow
```
pip install .
```
## Test torcharrow
```
python -m unittest
# or
pip install pytest
pytest torcharrow
```

## Type checking
```
pip install mypy
mypy torcharrow
```

## Code formatting

The code is auto-formatted with Black:

```
pip install black
black torcharrow
```

## Documentation
All documentation is available via Notebooks:
* [TorchArrow in 10 minutes (a tutorial)](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow10min.ipynb)
* [TorchArrow data pipes](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_data_pipes.ipynb)
* [TorchArrow state handling](https://github.com/facebookexternal/torchdata/blob/main/torcharrow/torcharrow_state.ipynb)
* TorchArrow multitargeting - TBD
* TorchArrow, Pandas, UPM and SQL: What's the difference - TBD
* TorchArrow, Design rationale - TBD

## Status
This directory supports rapid development. So expect frequent changes.

Still to be done:
* Add tabular.py as package in setup and not as code
* [DONE] How to do Multi-device targeting (See TorchArrow state handling notebook
* An example program analysis (types/PPF?)
* Add example UDFs
* Add Tensors as example UDTs
* [WORKS, example to be wriutten] Using Numba for Jitting


