# Install dependencies

```
pip3 install --pre torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/nightly/cu113
python setup.py develop
```

# Usage instructions


```
usage: run_benchmark.py [-h] [--dataset DATASET] [--model_name MODEL_NAME] [--batch_size BATCH_SIZE] [--device DEVICE] [--num_epochs NUM_EPOCHS] 
                        [--report_location REPORT_LOCATION] [--num_workers NUM_WORKERS] [--shuffle] [--dataloaderv DATALOADERV]
```

## Available metrics
* [x] Total time
* [x] Time per batch
* [x] Time per epoch
* [x] Precision over time
* [x] CPU Load
* [x] GPU Load
* [x] Memory usage
* [x] PyTorch profiler

## Additional profiling

```
pip install scalene
```
`scalene run_benchmark.py`


## Other benchmarks in the wild
* https://github.com/pytorch/kineto/blob/main/tb_plugin/examples/datapipe_example.py
* https://github.com/pytorch/text/tree/main/test/datasets
* https://github.com/pytorch/vision/tree/main/torchvision/prototype/datasets