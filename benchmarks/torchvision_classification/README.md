This folder contains the codes necessary to run the benchmark once the instances have been launched.

## Prerequisites

The `torchvision` library must be installed for this benchmark. The
[run_with_submitit.py](https://gist.github.com/datumbox/f98a0c995fd3078ffa6f1b07e33bfb69) file can be helpful when
submitting your job on a cluster.

## Downloading the datasets

The datasets are readily available for certain AWS cluster. The instructions for external cloud will be added here.

## Running the benchmark

The following command

```sh
PYTHONPATH=$PYTHONPATH:pwd python -u ./run_with_submitit.py \
  --ngpus 8 --nodes 1 \
  --model mobilenet_v3_large --epochs 5 --batch-size 128 --workers 12 \
  --ds-type dp --fs ontap --data-loader V2 --dataset
```

### Parameters

For `run_with_submitit.py` (with complete list is within that file):

- `ngpus` - number of GPUS to request on each node
- `nodes` - number of nodes to request

For `train.py` (with complete list is within that file):

- `data-loader` - DataLoader version; `V1` or `V2`
- `ds_type` - type of dataset; `dp` or `mapstyle` or `iterable` (focus on the first two)
- `model` - model to train (from `torchvision.models`)
- `epochs` - number of training epochs to run
- `batch-size` - images per gpu, the total batch size is `ngpus` x `batch_size`
- `workers` - number of DataLoader workers
- `fs` - file system; either `ontap` (faster and more common) or `fsx`
- `output-dir` - path to save outputs

## Datasets

### Vision

- Small:
  - Imagenette, ~10k
  - [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10),
    50k
  - tiny-imagenet, ~100k
- Medium:
  - [ImageNet](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet),
    1.3M, also known as ImageNet1K
- Large:
  - ImageNet22K, 14M
  - [LSUN](https://pytorch.org/vision/stable/generated/torchvision.datasets.LSUN.html#torchvision.datasets.LSUN), 60M

### Text

- Small:
  - [SST2](https://pytorch.org/text/stable/datasets.html#sst2)
- Medium:
  - [DBpedia](https://pytorch.org/text/stable/datasets.html#dbpedia)
- Large:
  - [EnWik9](https://pytorch.org/text/stable/datasets.html#enwik9)
