WebDataset
================

Format
---------------------------------------------

WebDataset stores deep learning as tar files, with the simple convention
that files that belong together and make up a training sample share
the same basename. WebDataset can read files from local disk or from
any pipe, which allows it to access files using common cloud object
stores. WebDataset can also read concatenated MsgPack and CBORs sources.

The WebDataset representation allows writing purely sequential I/O
pipelines for large scale deep learning. This is important for achieving
high I/O rates from local storage (3x-10x for local drives compared to
random access) and for using object stores and cloud storage for training.

The WebDataset format represents images, movies, audio, etc. in their
native file formats, making the creation of WebDataset format data as
easy as just creating a tar archive. Because of the way data is aligned,
WebDataset works well with block deduplication as well and aligns data
on predictable boundaries.

Standard tools can be.. code:: bash

.. code:: bash
   curl -s http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar | tar tf - | sed 10q

::

   e39871fd9fd74f55.jpg
   e39871fd9fd74f55.json
   f18b91585c4d3f3e.jpg
   f18b91585c4d3f3e.json
   ede6e66b2fb59aab.jpg
   ede6e66b2fb59aab.json
   ed600d57fcee4f94.jpg
   ed600d57fcee4f94.json
   ff47e649b23f446d.jpg
   ff47e649b23f446d.json

Related Projects
------------------

-  the new ``torchdata`` library in PyTorch will add native (built-in)
   support for WebDataset
-  the AIStore server provides high-speed storage, caching, and data
   transformation for WebDataset data
-  WebDataset training can be carried out directly against S3, GCS, and
   other cloud storage buckets
-  NVIDIA’s DALI library supports reading WebDataset format data
   directly
-  there is a companion project to read WebDataset data in Julia
-  the ``tarp`` command line program can be used for quick and easy
   dataset transformations of WebDataset data used for accessing and processing WebDataset-format files.

Using WebDataset in ``torchdata``
--------------------------------

This is a common pipeline for reading datasets in WebDataset format.
The same pipeline can be used to read many different datasets.
Following the last step, you can use any kind of augmentation or further
processing you like.

The three steps you usually want to customize are:

- ``filecache`` -- this can be omitted, or you can choose a different cache directory
- ``decode`` -- if you have other datatypes, you need to list decoders for them
- ``extract_keys`` -- extracts relevant parts of the sample and turns it into a tuple

.. code:: python
    import numpy as np
    import torchdata.datapipes as dp
    from itertools import islice
    from imageio import imread
    import io


    src = "https://storage.googleapis.com/nvdata-ocropus-bin/ia1-{000000..000033}.tar"

    ds = (
        dp.iter.IterableWrapper([src])
        .shardexpand()
        .shuffle()
        .popen()
        .filecache(cachedir="mycache", verbose=True)
        .load_from_tar(mode="r|")
        .decode(
            ("*.nrm.jpg", lambda s: imread(io.BytesIO(s.read()))),
            ("*.hocr", lambda s: s.read().decode("utf-8")),
        )
        .webdataset()
        .extract_keys("*.nrm.jpg", "*.hocr")
        .incshuffle(initial=5, buffer_size=10000)
    )

    print(list(islice(ds, 0, 10)))


.. code:: python
    ds = (
        # iterates through all source shard specs (only one in this case)
        dp.iter.IterableWrapper([src])

        # expand the {0000000..000033} notation in the shard specification
        .shardexpand()

        # shuffle the .tar files
        .shuffle()

        # open each of the .tar files in turn (this can open local or remote files)
        .popen()

        # cache the tar files locally so they don't have to be re-downloaded every time
        .filecache(cachedir="mycache", verbose=True)

        # extract the individual files from the tar files
        .load_from_tar(mode="r|")

        # decode files ending in .nrm.jpg and .hocr.
        # (we need the extra io.BytesIO because imread requires a seekable source)
        .decode(
            ("*.nrm.jpg", lambda s: imread(io.BytesIO(s.read()))),
            ("*.hocr", lambda s: s.read().decode("utf-8")),
        )

        # group together files in the tar file based on WebDataset conventions
        # this returns a stream of dictionaries
        .webdataset()

        # turn the dictionaries into tuples by extracting the sample key and
        # anything that matches one of the image formats
        .extract_keys("*.nrm.jpg", "*.hocr")

        # incrementally shuffle the resulting training samples
        .incshuffle(initial=5, buffer_size=10000)
    )

    print(list(islice(ds, 0, 10)))


Available Functionality
--------------------------------


The following filters are particularly useful with WebDataset-style data, but
they can be used with other kinds of datasets too:

- WebDataset / .webdataset -- group files in stream into samples based on basename
- ShardExpander / .shardexpand -- expand brace notations for file shards
- FileDecoder / .decode -- apply file decoders based on matching glob patterns / extensions
- PipeOpener / .popen -- reads local and remote datasets using command line programs
- RenameKeys / .rename_keys -- rename samples in a dictionary by matching keys against glob patterns
- ExtractKeys / .extract_keys -- extract sample fields and return tuple based on glob patterns
- IncrementalShuffler / .incshuffle -- inline shuffling of shards and samples with low startup latency
- FileCache / .filecache -- cache files / shards in a local data directory
