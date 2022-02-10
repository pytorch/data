Tutorial
================

Implementing a Custom DataPipe
---------------------------------------------
As a guiding example, let's implement an ``IterDataPipe`` that applies a callable to the input iterator. For
``MapDataPipe``, take a look at the
`map <https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes/map>`_
folder for examples, and follow the steps below for the ``__getitem__`` method instead of  the ``__iter__`` method.

Naming
^^^^^^^^^^^^^^^^^^
The naming convention for ``DataPipe`` is "Operation"-er, followed by ``IterDataPipe`` or ``MapDataPipe``, as each
DataPipe is essentially a container to apply an operation to data yielded from a source ``DataPipe``. For succinctness,
we alias to just "Operation-er" in **init** files. For our ``IterDataPipe`` example, we'll name the module
``MapperIterDataPipe`` and alias it as ``iter.Mapper`` under ``torchdata.datapipes``.


Constructor
^^^^^^^^^^^^^^^^^^

DataSets are now generally constructed as stacks of ``DataPipes``, so each ``DataPipe`` typically takes a
source ``DataPipe`` as its first argument.

.. code:: python

    from torchdata.datapipes.iter import IterDataPipe

    class MapperIterDataPipe(IterDataPipe):
        def __init__(self, source_dp: IterDataPipe, fn) -> None:
            super().__init__()
            self.source_dp = source_dp
            self.fn = fn

Note:

- Avoid loading data from the source DataPipe in ``__init__`` function, in order to support lazy data loading and save
  memory.

- If ``IterDataPipe`` instance holds data in memory, please be ware of the in-place modification of data. When second
  iterator is created from the instance, the data may have already changed. Please take ``IterableWrapper``
  `class <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/datapipes/iter/utils.py>`_
  as reference to ``deepcopy`` data for each iterator.

- Avoid variables names that are taken by the functional names of existing DataPipes. For instance, ``.filter`` is
  is functional name that can be used to invoke ``FilterIterDataPipe``. Having a variable named ``filter`` inside
  another ``IterDataPipe`` can lead to confusion.


Iterator
^^^^^^^^^^^^^^^^^^
For ``IterDataPipes``, an ``__iter__`` function is needed to consume data from the source ``IterDataPipe`` then
apply the operation over the data before ``yield``.

.. code:: python

    class MapperIterDataPipe(IterDataPipe):
        # ... See __init__() defined above

        def __iter__(self):
            for d in self.dp:
                yield self.fn(d)

Length
^^^^^^^^^^^^^^^^^^
In many cases, as in our ``MapperIterDataPipe`` example, the ``__len__`` method of a DataPipe returns the length of the
source DataPipe.

.. code:: python

    class MapperIterDataPipe(IterDataPipe):
        # ... See __iter__() defined above

        def __len__(self):
            return len(self.dp)

However, note that ``__len__`` is optional for ``IterDataPipe`` and often inadvisable. For ``CSVParserIterDataPipe``
in the using DataPipes section below, ``__len__`` is not implemented because the number of rows in each file
is unknown before loading it. In some special cases, ``__len__`` can be made to either return an integer or raise
an Error depending on the input. In those cases, the Error must be a ``TypeError`` to support Python's
build-in functions like ``list(dp)``.

Registering DataPipes with the functional API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each DataPipe can be registered to support functional invocation using the decorator ``functional_datapipe``.

.. code:: python

    @functional_datapipe("map")
    class MapperIterDataPipe(IterDataPipe):
       # ...

The stack of DataPipes can then be constructed in functional form:

.. code:: python

    import torch.utils.data.datapipes as dp

    datapipes1 = dp.iter.FileOpener(['a.file', 'b.file']).map(fn=decoder).shuffle().batch(2)
    datapipes2 = dp.iter.FileOpener(['a.file', 'b.file'])
    datapipes2 = dp.iter.Mapper(datapipes2, fn=decoder)
    datapipes2 = dp.iter.Shuffler(datapipes2)
    datapipes2 = dp.iter.Batcher(datapipes2, 2)

In the above example, ``datapipes1`` and ``datapipes2`` represent the exact same stack of ``IterDataPipe``\s.

Using DataPipes
---------------------------------------------

For a complete example, suppose that we want to load data from CSV files with the following steps:

- List all CSV files in a directory
- Load CSV files
- Parse CSV file and yield rows

To support the above pipeline, ``CSVParser`` is registered with the functional name ``parse_csv`` to consume
file streams, parse the CSV contents, and return one parsed line at a time.

.. code:: python

    @functional_datapipe("parse_csv")
    class CSVParserIterDataPipe(IterDataPipe):
        def __init__(self, dp, **fmtparams) -> None:
            self.dp = dp
            self.fmtparams = fmtparams

        def __iter__(self) -> Iterator[Union[Str_Or_Bytes, Tuple[str, Str_Or_Bytes]]]:
            for path, file in self.source_datapipe:
                stream = self._helper.skip_lines(file)
                stream = self._helper.strip_newline(stream)
                stream = self._helper.decode(stream)
                yield from self._helper.return_path(stream, path=path)  # Returns 1 line at a time as List[str or bytes]

Then, the pipeline can be assembled as the following:

.. code:: python

    import torch.utils.data.datapipes as dp

    FOLDER = 'path/2/csv/folder'
    datapipe = dp.iter.FileLister([FOLDER]).filter(fn=lambda filename: filename.endswith('.csv'))
    datapipe = dp.iter.FileOpener(datapipe, mode='rt')
    datapipe = datapipe.parse_csv(delimiter=',')

    for d in datapipe: # Start loading data
         pass


Working with DataLoader
---------------------------------------------


Single-Process Data Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Multi-Process Data Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
