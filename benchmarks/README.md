# Install dependencies

```
pip install -r benchmarks/requirements.txt
python setup.py develop
```

# Usage instructions

```
usage: run_benchmark.py [-h] [--dataset DATASET] [--model_name MODEL_NAME] [--batch_size BATCH_SIZE] [--device DEVICE] [--num_epochs NUM_EPOCHS]
                        [--report_location REPORT_LOCATION] [--num_workers NUM_WORKERS] [--shuffle] [--dataloaderv DATALOADERV]
```

## Available metrics

- [x] Total time
- [x] Time per batch
- [x] Time per epoch
- [x] Precision over time
- [x] CPU Load
- [x] GPU Load
- [x] Memory usage

## Additional profiling

The PyTorch profiler doesn't work quite well with `torchdata` for now https://github.com/pytorch/kineto/issues/609 but
there are other good options like `py-spy` or `scalene` which could be used like so `profiler_name run_benchmark.py`

## Other benchmarks in the wild

- https://github.com/pytorch/kineto/blob/main/tb_plugin/examples/datapipe_example.py
- https://github.com/pytorch/text/tree/main/test/datasets
- https://github.com/pytorch/vision/tree/main/torchvision/prototype/datasets
