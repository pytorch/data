Examples
================

.. currentmodule:: examples

In this section, you will find the data loading implementations (using DataPipes) of various
popular datasets across different research domains. Some of the examples are implements by the PyTorch team and the
implementation codes are maintained within PyTorch libraries. Others are created by members of the PyTorch community.

Audio
-----------

LibriSpeech
^^^^^^^^^^^^^^^^^^^^^^^^^^

`LibriSpeech dataset <https://www.openslr.org/12/>`_ is corpus of approximately 1000 hours of 16kHz read
English speech. Here is the
`DataPipe implementation of LibriSpeech <https://github.com/pytorch/data/blob/main/examples/audio/librispeech.py>`_
to load the data.

Text
-----------

Amazon Review Polarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Amazon reviews dataset contains reviews from Amazon. Its purpose is to train text/sentiment classification models.
In our DataPipe
`implementation of the dataset <https://github.com/pytorch/data/blob/main/examples/text/amazonreviewpolarity.py>`_,
we described every step with detailed comments to help you understand what each DataPipe is doing. We recommend
having a look at this example.


IMDB
^^^^^^^^^^^^^^^^^^^^^^^^^^
This is a `large movie review dataset <http://ai.stanford.edu/~amaas/data/sentiment/>`_ for binary sentiment
classification containing 25,000 highly polar movie reviews for training and 25,00 for testing. Here is the
`DataPipe implementation to load the data <https://github.com/pytorch/data/blob/main/examples/text/imdb.py>`_.


SQuAD
^^^^^^^^^^^^^^^^^^^^^^^^^^
`SQuAD (Stanford Question Answering Dataset) <https://rajpurkar.github.io/SQuAD-explorer/>`_ is a dataset for
reading comprehension. It consists of a list of questions by crowdworkers on a set of Wikipedia articles. Here are the
DataPipe implementations for `version 1.1 <https://github.com/pytorch/data/blob/main/examples/text/squad1.py>`_
is here and `version 2.0 <https://github.com/pytorch/data/blob/main/examples/text/squad2.py>`_.

Additional Datasets in TorchText
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In a separate PyTorch domain library `TorchText <https://github.com/pytorch/text>`_, you will find some of the most
popular datasets in the NLP field implemented as loadable datasets using DataPipes. You can find
all of those `NLP datasets here <https://github.com/pytorch/text/tree/main/torchtext/datasets>`_.


Vision
-----------

Caltech 101
^^^^^^^^^^^^^^^^^^^^^^^^^^
The `Caltech 101 dataset <https://data.caltech.edu/records/20086>`_ contains pictures of objects
belonging to 101 categories. Here is the
`DataPipe implementation of Caltech 101 <https://github.com/pytorch/data/blob/main/examples/vision/caltech101.py>`_.

Caltech 256
^^^^^^^^^^^^^^^^^^^^^^^^^^
The `Caltech 256 dataset <https://data.caltech.edu/records/20087>`_ contains 30607 images
from 256 categories. Here is the
`DataPipe implementation of Caltech 256 <https://github.com/pytorch/data/blob/main/examples/vision/caltech256.py>`_.

CamVid - Semantic Segmentation (community example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The `Cambridge-driving Labeled Video Database (CamVid) <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>`_ is a collection of videos with object class semantic 
labels, complete with metadata. The database provides ground truth labels that associate each pixel with one of 32 
semantic classes. Here is a
`DataPipe implementation of CamVid 
<https://github.com/tcapelle/torchdata/blob/main/01_Camvid_segmentation_with_datapipes.ipynb>`_
created by our community.

laion2B-en-joined
^^^^^^^^^^^^^^^^^^^^^^
The `laion2B-en-joined dataset <https://huggingface.co/datasets/laion/laion2B-en-joined>`_ is a subset of the `LAION-5B dataset <https://laion.ai/blog/laion-5b/>`_ containing english captions, URls pointing to images,
and  other metadata. It contains around 2.32 billion entries.
Currently (February 2023) around 86% of the URLs still point to valid images. Here is a `DataPipe implementation of laion2B-en-joined
<https://github.com/pytorch/data/blob/main/examples/vision/laion5b.py>`_ that filters out unsafe images and images with watermarks and loads the images from the URLs.

Additional Datasets in TorchVision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In a separate PyTorch domain library `TorchVision <https://github.com/pytorch/vision>`_, you will find some of the most
popular datasets in the computer vision field implemented as loadable datasets using DataPipes. You can find all of
those `vision datasets here <https://github.com/pytorch/vision/tree/main/torchvision/prototype/datasets/_builtin>`_.

Note that these implementations are currently in the prototype phase, but they should be fully supported
in the coming months. Nonetheless, they demonstrate the different ways DataPipes can be used for data loading.

Recommender System
---------------------------------

Criteo 1TB Click Logs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The `Criteo dataset <https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset>`_ contains feature values
and click feedback for millions of display advertisements. It aims to benchmark algorithms for
click through rate (CTR) prediction. You can find a prototype stage implementation of the
`dataset with DataPipes in TorchRec <https://github.com/pytorch/torchrec/blob/main/torchrec/datasets/criteo.py>`_.

Graphs, Meshes and Point Clouds
-------------------------------

TigerGraph (community example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TigerGraph is a scalable graph data platform for AI and ML. You can find an `implementation <https://github.com/TigerGraph-DevLabs/torchdata_tutorial/blob/main/torchdata_example.ipynb>`_ of graph feature engineering and machine learning with DataPipes in TorchData and data stored in a TigerGraph database, which includes computing PageRank scores in-database, pulling graph data and features with multiple DataPipes, and training a neural network using graph features in PyTorch. 

MoleculeNet (community example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`MoleculeNet <https://moleculenet.org/>`_ is a benchmark specially designed for testing machine learning methods of
molecular properties. You can find an implementation of the
`HIV dataset with DataPipes in PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/datapipe.py>`_,
which includes converting SMILES strings into molecular graph representations.

Princeton ModelNet (community example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Princeton ModelNet project provides a comprehensive and clean collection of 3D CAD models across various object types.
You can find an implementation of the
`ModelNet10 dataset with DataPipes in PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/datapipe.py>`_,
which includes reading in meshes via `meshio <https://github.com/nschloe/meshio>`_, and sampling of points from object surfaces and dynamic
graph generation via `PyG's functional transformations <https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html>`_.

Timeseries
---------------------------------

Custom DataPipe for Timeseries rolling window (community example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Implementing a rolling window custom `DataPipe` for timeseries forecasting tasks.
Here is the
`DataPipe implementation of a rolling window 
<https://github.com/tcapelle/torchdata/blob/main/02_Custom_timeseries_datapipe.ipynb>`_.


Using AIStore
-------------------------

Caltech 256 and Microsoft COCO (community example)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Listing and loading data from AIS buckets (buckets that are not 3rd party backend-based) and remote cloud buckets (3rd party 
backend-based cloud buckets) using `AISFileLister <https://pytorch.org/data/main/generated/torchdata.datapipes.iter.AISFileLister.html#aisfilelister>`_ and `AISFileLoader <https://pytorch.org/data/main/generated/torchdata.datapipes.iter.AISFileLoader.html#torchdata.datapipes.iter.AISFileLoader>`_.

Here is an `example which uses AISIO DataPipe <https://github.com/pytorch/data/blob/main/examples/aistore/aisio_usage_example.ipynb>`_ for the `Caltech-256 Object Category Dataset <https://data.caltech.edu/records/20087>`_ containing 256 object categories and a total 
of 30607 images stored on an AIS bucket and the `Microsoft COCO Dataset <https://cocodataset.org/#home>`_ which has 330K images with over 200K 
labels of more than 1.5 million object instances across 80 object categories stored on Google Cloud.
