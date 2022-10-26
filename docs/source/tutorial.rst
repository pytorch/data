Tutorial
================

Using DataPipes
---------------------------------------------

Suppose that we want to load data from CSV files with the following steps:

- List all CSV files in a directory
- Load CSV files
- Parse CSV file and yield rows
- Split our dataset into training and validation sets

There are a few `built-in DataPipes <torchdata.datapipes.iter.html>`_ that can help us with the above operations.

- ``FileLister`` - `lists out files in a directory <generated/torchdata.datapipes.iter.FileLister.html>`_
- ``Filter`` - `filters the elements in DataPipe based on a given
  function <generated/torchdata.datapipes.iter.Filter.html>`_
- ``FileOpener`` - `consumes file paths and returns opened file
  streams <generated/torchdata.datapipes.iter.FileOpener.html>`_
- ``CSVParser`` - `consumes file streams, parses the CSV contents, and returns one parsed line at a
  time <generated/torchdata.datapipes.iter.CSVParser.html>`_
- ``RandomSplitter`` - `randomly split samples from a source DataPipe into
  groups <generated/torchdata.datapipes.iter.RandomSplitter.html>`_

As an example, the source code for ``CSVParser`` looks something like this:

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

As mentioned in a different section, DataPipes can be invoked using their functional forms (recommended) or their
class constructors. A pipeline can be assembled as the following:

.. code:: python

    import torchdata.datapipes as dp

    FOLDER = 'path/2/csv/folder'
    datapipe = dp.iter.FileLister([FOLDER]).filter(filter_fn=lambda filename: filename.endswith('.csv'))
    datapipe = dp.iter.FileOpener(datapipe, mode='rt')
    datapipe = datapipe.parse_csv(delimiter=',')
    N_ROWS = 10000  # total number of rows of data
    train, valid = datapipe.random_split(total_length=N_ROWS, weights={"train": 0.5, "valid": 0.5}, seed=0)

    for x in train:  # Iterating through the training dataset
        pass

    for y in valid:  # Iterating through the validation dataset
        pass

You can find the full list of built-in `IterDataPipes here <torchdata.datapipes.iter.html>`_ and
`MapDataPipes here <torchdata.datapipes.map.html>`_.

Working with DataLoader
---------------------------------------------

In this section, we will demonstrate how you can use DataPipe with ``DataLoader``.
For the most part, you should be able to use it just by passing ``dataset=datapipe`` as an input arugment
into the ``DataLoader``. For detailed documentation related to ``DataLoader``,
please visit `this page <https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading>`_.


For this example, we will first have a helper function that generates some CSV files with random label and data.

.. code:: python

    import csv
    import random

    def generate_csv(file_label, num_rows: int = 5000, num_features: int = 20) -> None:
        fieldnames = ['label'] + [f'c{i}' for i in range(num_features)]
        writer = csv.DictWriter(open(f"sample_data{file_label}.csv", "w", newline=''), fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_rows):
            row_data = {col: random.random() for col in fieldnames}
            row_data['label'] = random.randint(0, 9)
            writer.writerow(row_data)

Next, we will build our DataPipes to read and parse through the generated CSV files. Note that we prefer to have
pass defined functions to DataPipes rather than lambda functions because the formers are serializable with `pickle`.

.. code:: python

    import numpy as np
    import torchdata.datapipes as dp

    def filter_for_data(filename):
        return "sample_data" in filename and filename.endswith(".csv")

    def row_processer(row):
        return {"label": np.array(row[0], np.int32), "data": np.array(row[1:], dtype=np.float64)}

    def build_datapipes(root_dir="."):
        datapipe = dp.iter.FileLister(root_dir)
        datapipe = datapipe.filter(filter_fn=filter_for_data)
        datapipe = datapipe.open_files(mode='rt')
        datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1)
        # Shuffle will happen as long as you do NOT set `shuffle=False` later in the DataLoader
        datapipe = datapipe.shuffle()
        datapipe = datapipe.map(row_processer)
        return datapipe

Lastly, we will put everything together in ``'__main__'`` and pass the DataPipe into the DataLoader. Note that
if you choose to use ``Batcher`` while setting ``batch_size > 1`` for DataLoader, your samples will be
batched more than once. You should choose one or the other.

.. code:: python

    from torch.utils.data import DataLoader

    if __name__ == '__main__':
        num_files_to_generate = 3
        for i in range(num_files_to_generate):
            generate_csv(file_label=i, num_rows=10, num_features=3)
        datapipe = build_datapipes()
        dl = DataLoader(dataset=datapipe, batch_size=5, num_workers=2)
        first = next(iter(dl))
        labels, features = first['label'], first['data']
        print(f"Labels batch shape: {labels.size()}")
        print(f"Feature batch shape: {features.size()}")
        print(f"{labels = }\n{features = }")
        n_sample = 0
        for row in iter(dl):
            n_sample += 1
        print(f"{n_sample = }")

The following statements will be printed to show the shapes of a single batch of labels and features.

.. code::

    Labels batch shape: torch.Size([5])
    Feature batch shape: torch.Size([5, 3])
    labels = tensor([8, 9, 5, 9, 7], dtype=torch.int32)
    features = tensor([[0.2867, 0.5973, 0.0730],
            [0.7890, 0.9279, 0.7392],
            [0.8930, 0.7434, 0.0780],
            [0.8225, 0.4047, 0.0800],
            [0.1655, 0.0323, 0.5561]], dtype=torch.float64)
    n_sample = 12

The reason why ``n_sample = 12`` is because ``ShardingFilter`` (``datapipe.sharding_filter()``) was not used, such that
each worker will independently return all samples. In this case, there are 10 rows per file and 3 files, with a
batch size of 5, that gives us 6 batches per worker. With 2 workers, we get 12 total batches from the ``DataLoader``.

In order for DataPipe sharding to work with ``DataLoader``, we need to add the following.

.. code:: python

    def build_datapipes(root_dir="."):
        datapipe = ...
        # Add the following line to `build_datapipes`
        # Note that it is somewhere after `Shuffler` in the DataPipe line, but before expensive operations
        datapipe = datapipe.sharding_filter()
        return datapipe

When we re-run, we will get:

.. code::

    ...
    n_sample = 6

Note:

- Place ``ShardingFilter`` (``datapipe.sharding_filter``) as early as possible in the pipeline, especially before expensive
  operations such as decoding, in order to avoid repeating these expensive operations across worker/distributed processes.
- For the data source that needs to be sharded, it is crucial to add ``Shuffler`` before ``ShardingFilter``
  to ensure data are globally shuffled before splitted into shards. Otherwise, each worker process would
  always process the same shard of data for all epochs. And, it means each batch would only consist of data
  from the same shard, which leads to low accuracy during training. However, it doesn't apply to the data
  source that has already been sharded for each multi-/distributed process, since ``ShardingFilter`` is no
  longer required to be presented in the pipeline.
- There may be cases where placing ``Shuffler`` earlier in the pipeline lead to worse performance, because some
  operations (e.g. decompression) are faster with sequential reading. In those cases, we recommend decompressing
  the files prior to shuffling (potentially prior to any data loading).


You can find more DataPipe implementation examples for various research domains `on this page <examples.html>`_.


Implementing a Custom DataPipe
---------------------------------------------
Currently, we already have a large number of built-in DataPipes and we expect them to cover most necessary
data processing operations. If none of them supports your need, you can create your own custom DataPipe.

As a guiding example, let us implement an ``IterDataPipe`` that applies a callable to the input iterator. For
``MapDataPipe``, take a look at the
`map <https://github.com/pytorch/pytorch/tree/master/torch/utils/data/datapipes/map>`_
folder for examples, and follow the steps below for the ``__getitem__`` method instead of  the ``__iter__`` method.

Naming
^^^^^^^^^^^^^^^^^^
The naming convention for ``DataPipe`` is "Operation"-er, followed by ``IterDataPipe`` or ``MapDataPipe``, as each
DataPipe is essentially a container to apply an operation to data yielded from a source ``DataPipe``. For succinctness,
we alias to just "Operation-er" in **init** files. For our ``IterDataPipe`` example, we'll name the module
``MapperIterDataPipe`` and alias it as ``iter.Mapper`` under ``torchdata.datapipes``.

For the functional method name, the naming convention is ``datapipe.<operation>``. For instance,
the functional method name of ``Mapper`` is ``map``, such that it can be invoked by ``datapipe.map(...)``.


Constructor
^^^^^^^^^^^^^^^^^^

DataSets are now generally constructed as stacks of ``DataPipes``, so each ``DataPipe`` typically takes a
source ``DataPipe`` as its first argument. Here is a simplified version of `Mapper` as an example:

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
  the functional name that can be used to invoke ``FilterIterDataPipe``. Having a variable named ``filter`` inside
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

The stack of DataPipes can then be constructed using their functional forms (recommended) or class constructors:

.. code:: python

    import torchdata.datapipes as dp

    # Using functional form (recommended)
    datapipes1 = dp.iter.FileOpener(['a.file', 'b.file']).map(fn=decoder).shuffle().batch(2)
    # Using class constructors
    datapipes2 = dp.iter.FileOpener(['a.file', 'b.file'])
    datapipes2 = dp.iter.Mapper(datapipes2, fn=decoder)
    datapipes2 = dp.iter.Shuffler(datapipes2)
    datapipes2 = dp.iter.Batcher(datapipes2, 2)

In the above example, ``datapipes1`` and ``datapipes2`` represent the exact same stack of ``IterDataPipe``\s. We
recommend using the functional form of DataPipes.

Working with Cloud Storage Providers
---------------------------------------------

In this section, we show examples accessing AWS S3, Google Cloud Storage, and Azure Cloud Storage with built-in ``fsspec`` DataPipes.
Although only those two providers are discussed here, with additional libraries, ``fsspec`` DataPipes
should allow you to connect with other storage systems as well (`list of known
implementations <https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations>`_).

Let us know on GitHub if you have a request for support for other cloud storage providers,
or you have code examples to share with the community.

Accessing AWS S3 with ``fsspec`` DataPipes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This requires the installation of the libraries ``fsspec``
(`documentation <https://filesystem-spec.readthedocs.io/en/latest/>`_) and ``s3fs``
(`s3fs GitHub repo <https://github.com/fsspec/s3fs>`_).

You can list out the files within a S3 bucket directory by passing a path that starts
with ``"s3://BUCKET_NAME"`` to
`FSSpecFileLister <generated/torchdata.datapipes.iter.FSSpecFileLister.html>`_ (``.list_files_by_fsspec(...)``).

.. code:: python

    from torchdata.datapipes.iter import IterableWrapper

    dp = IterableWrapper(["s3://BUCKET_NAME"]).list_files_by_fsspec()

You can also open files using `FSSpecFileOpener <generated/torchdata.datapipes.iter.FSSpecFileOpener.html>`_
(``.open_files_by_fsspec(...)``) and stream them
(if supported by the file format).

Note that you can also provide additional parameters via
the argument ``kwargs_for_open``. This can be useful for purposes such as accessing specific
bucket version, which you can do so by passing in ``{version_id: 'SOMEVERSIONID'}`` (more `details
about S3 bucket version awareness <https://s3fs.readthedocs.io/en/latest/#bucket-version-awareness>`_
by ``s3fs``). The supported arguments vary by the (cloud) file system that you are accessing.

In the example below, we are streaming the archive by using
`TarArchiveLoader <generated/torchdata.datapipes.iter.TarArchiveLoader.html#>`_ (``.load_from_tar(mode="r|")``),
in contrast with the usual ``mode="r:"``. This allows us to begin processing data inside the archive
without downloading the whole archive into memory first.

.. code:: python

    from torchdata.datapipes.iter import IterableWrapper
    dp = IterableWrapper(["s3://BUCKET_NAME/DIRECTORY/1.tar"])
    dp = dp.open_files_by_fsspec(mode="rb", anon=True).load_from_tar(mode="r|") # Streaming version
    # The rest of data processing logic goes here


Finally, `FSSpecFileSaver <generated/torchdata.datapipes.iter.FSSpecSaver.html>`_
is also available for writing data to cloud.

Accessing Google Cloud Storage (GCS) with ``fsspec`` DataPipes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This requires the installation of the libraries ``fsspec``
(`documentation <https://filesystem-spec.readthedocs.io/en/latest/>`_) and ``gcsfs``
(`gcsfs GitHub repo <https://github.com/fsspec/gcsfs>`_).

You can list out the files within a GCS bucket directory by specifying a path that starts
with ``"gcs://BUCKET_NAME"``. The bucket name in the example below is ``uspto-pair``.

.. code:: python

    from torchdata.datapipes.iter import IterableWrapper

    dp = IterableWrapper(["gcs://uspto-pair/"]).list_files_by_fsspec()
    print(list(dp))
    # ['gcs://uspto-pair/applications', 'gcs://uspto-pair/docs', 'gcs://uspto-pair/prosecution-history-docs']

Here is an example of loading a zip file ``05900035.zip`` from a bucket named ``uspto-pair`` inside the
directory ``applications``.

.. code:: python

    from torchdata.datapipes.iter import IterableWrapper

    dp = IterableWrapper(["gcs://uspto-pair/applications/05900035.zip"]) \
            .open_files_by_fsspec(mode="rb") \
            .load_from_zip()
    # Logic to process those archive files comes after
    for path, filestream in dp:
        print(path, filestream)
    # gcs:/uspto-pair/applications/05900035.zip/05900035/README.txt, StreamWrapper<...>
    # gcs:/uspto-pair/applications/05900035.zip/05900035/05900035-address_and_attorney_agent.tsv, StreamWrapper<...>
    # gcs:/uspto-pair/applications/05900035.zip/05900035/05900035-application_data.tsv, StreamWrapper<...>
    # gcs:/uspto-pair/applications/05900035.zip/05900035/05900035-continuity_data.tsv, StreamWrapper<...>
    # gcs:/uspto-pair/applications/05900035.zip/05900035/05900035-transaction_history.tsv, StreamWrapper<...>

Accessing Azure Blob storage with ``fsspec`` DataPipes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This requires the installation of the libraries ``fsspec``
(`documentation <https://filesystem-spec.readthedocs.io/en/latest/>`_) and ``adlfs``
(`adlfs GitHub repo <https://github.com/fsspec/adlfs>`_).
You can access data in Azure Data Lake Storage Gen2 by providing URIs staring with ``abfs://``. 
For example,
`FSSpecFileLister <generated/torchdata.datapipes.iter.FSSpecFileLister.html>`_ (``.list_files_by_fsspec(...)``) 
can be used to list files in a directory in a container:

.. code:: python

    from torchdata.datapipes.iter import IterableWrapper

    storage_options={'account_name': ACCOUNT_NAME, 'account_key': ACCOUNT_KEY}
    dp = IterableWrapper(['abfs://CONTAINER/DIRECTORY']).list_files_by_fsspec(**storage_options)
    print(list(dp))
    # ['abfs://container/directory/file1.txt', 'abfs://container/directory/file2.txt', ...]

You can also open files using `FSSpecFileOpener <generated/torchdata.datapipes.iter.FSSpecFileOpener.html>`_
(``.open_files_by_fsspec(...)``) and stream them
(if supported by the file format).

Here is an example of loading a CSV file ``ecdc_cases.csv`` from a public container inside the
directory ``curated/covid-19/ecdc_cases/latest``, belonging to account ``pandemicdatalake``.

.. code:: python

    from torchdata.datapipes.iter import IterableWrapper
    dp = IterableWrapper(['abfs://public/curated/covid-19/ecdc_cases/latest/ecdc_cases.csv']) \
            .open_files_by_fsspec(account_name='pandemicdatalake') \
            .parse_csv()
    print(list(dp)[:3])
    # [['date_rep', 'day', ..., 'iso_country', 'daterep'], 
    # ['2020-12-14', '14', ..., 'AF', '2020-12-14'],
    # ['2020-12-13', '13', ..., 'AF', '2020-12-13']]

If necessary, you can also access data in Azure Data Lake Storage Gen1 by using URIs staring with 
``adl://`` and ``abfs://``, as described in `README of adlfs repo <https://github.com/fsspec/adlfs/blob/main/README.md>`_
