import sys
import logging
import subprocess

import torchvision
import torch
import transformers
from torchvision.prototype.datasets import load as loadpipe
import torch.nn.functional as F
from torchvision import transforms
import time
import torch.optim as optim
import torch.profiler

# Relative imports
from args import arg_parser
from benchmarks.datasets import prepare_gtsrb_dataset
from utils import init_fn
from datasets import prepare_gtsrb_datapipe
from trainers import train
from report import create_report

logging.basicConfig(filename='example.log', level=logging.DEBUG)


dataset, ispipe, model_name, batch_size, device, num_epochs, num_workers, shuffle, dataloaderv = arg_parser()

if device.startswith("cuda"):
    nvidiasmi = subprocess.check_output("nvidia-smi", shell=True, text=True)
    logging.debug(nvidiasmi)

lscpu = subprocess.check_output("lscpu", shell=True, text=True)
logging.debug(lscpu)


if dataloaderv == 1:
    from torch.utils.data import DataLoader
elif dataloaderv == 2:
    from torch.utils.data.dataloader_experimental import DataLoader2 as DataLoader
else:
    raise(f"dataloaderv{dataloaderv} is not a valid option")

# Download model
model_map = {
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "mobilenet_v3_large" : torchvision.models.mobilenet_v3_large,
    "transformerencoder" : torch.nn.TransformerEncoder,
    "bert-base" : transformers.BertModel,

}

model = model_map[model_name]().to(torch.device(device))

# setup data pipe
if dataset == "gtsrb":
    if ispipe:
        dp = loadpipe(dataset, split="train")
        logging.debug(f"data format before preprocessing is {next(iter(dp))}")

        dp = prepare_gtsrb_datapipe(batch_size, device, dp)
        logging.debug(f"data format after preprocessing is \n {next(iter(dp))}\n")

    else:
        # No further preprocessing needed this returns a tuple of Images and labels as ints
        # Do I need to do batching and collation manually?
        ds = torchvision.datasets.GTSRB(root=".",split="train", download=True)


else:
    print(f"{dataset} not supported yet")


print(f"batch size {batch_size}")
print(f"Dataset name {dp}")
print(f"Dataset length {len(dp)}")

# Setup data loader

data = dp if dp else ds

if num_workers == 1:
    dl = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

# Shuffle won't work in distributed yet
else:
    dl = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=init_fn, multiprocessing_context="spawn")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



total_start = time.time()
per_epoch_durations = []
batch_durations = []



with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='datapipe0'),
    schedule=torch.profiler.schedule(wait=1,warmup=1,active=2),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    with_stack=True,
    with_modules=True
) as p:

    train(num_epochs, model, dl, per_epoch_durations, batch_durations, criterion, optimizer, p)

    total_end = time.time()
    total_duration = total_end - total_start

# TODO: Make this output some human readable markdown file
create_report(per_epoch_durations, batch_durations, total_duration)


