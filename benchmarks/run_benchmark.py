import argparse
import sys
import torchvision
import torch
import transformers

from torchvision.prototype.datasets import load
import torch.nn.functional as F
from torchvision import transforms
import time
from statistics import mean
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

## Arg parsing
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="gtsrb", help="The name of the dataset")
parser.add_argument("--model_name", type=str, default="resnext50_32x4d", help="The name of the model")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--device", type=str, default="cuda:0", help="Options are are cpu or cuda:0")
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--report_location", type=str, default="./report.md", help="The location where the generated report will be stored")
parser.add_argument("--num_workers", type=int, default=1, help="Number of dataloader workers")
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--dataloaderv", type=int, default=1)

args = parser.parse_args()
dataset = args.dataset
model_name = args.model_name
batch_size = args.batch_size
device = args.device
num_epochs = args.num_epochs
report_location = args.report_location
num_workers = args.num_workers
shuffle = args.shuffle
dataloaderv = args.dataloaderv

if dataloaderv == 1:
    from torch.utils.data import DataLoader
elif dataloaderv == 2:
    from torch.utils.data.dataloader_experimental import DataLoader2 as DataLoader
else:
    raise(f"dataloaderv{dataloaderv} is not a valid option")

# Util function for multiprocessing
def init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    num_workers = info.num_workers
    datapipe = info.dataset
    torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


# Download model
model_map = {
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "mobilenet_v3_large" : torchvision.models.mobilenet_v3_large,
    "transformerencoder" : torch.nn.TransformerEncoder,
    "bert-base" : transformers.BertModel,

}

model = model_map[model_name]().to(torch.device(device))

# setup data pipe
if model_name in ["resnext50_32x4d", "mobilenet_v3_large"]:
    dp = load(dataset, split="train")

else:
    print(f"{model} not supported yet")

print(f"batch size {batch_size}")
print(f"Dataset name {dp}")
print(f"Dataset length {len(dp)}")

# Datapipe format
# print(f"data format before preprocessing is {next(iter(dp))}")

if dataset == "gtsrb":
    def transform(img):
        t= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(96,98)),
            # transforms.reshape(64,3,7,7),
            transforms.ToTensor()]
        )
        return t(img).to(torch.device(device))

    def str_to_list(str):
        l = []
        for char in str:
            l.append(int(char))
        return l

    # Filter out bounding box and path to image
    dp = dp.map(lambda sample : {"image" : sample["image"], "label" : sample["label"]})

    # Apply image preprocessing
    dp = dp.map(lambda sample : transform(sample.decode()), input_col="image")
    dp = dp.map(lambda sample : torch.tensor(str_to_list(sample.to_categories())).to(torch.device(device)), input_col="label")

    # TODO: Missing a collation

    # Batch
    dp = dp.batch(batch_size)
    
# Datapipe format after preprocessing
# print(f"data format after preprocessing is \n {next(iter(dp))}\n")

# Setup data loader
if num_workers == 1:
    dl = DataLoader(dataset=dp, batch_size=batch_size, shuffle=shuffle)

# Shuffle won't work in distributed yet
else:
    dl = DataLoader(dataset=dp, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=init_fn, multiprocessing_context="spawn")


total_start = time.time()
per_epoch_durations = []
batch_durations = []

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=trace_handler
) as p:

    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0
        for i, elem in enumerate(dl):
            batch_start = time.time()

            labels = torch.argmax(elem[0]["label"], dim=1)      
            optimizer.zero_grad()
            outputs = model(elem[0]["image"])
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

            batch_end = time.time()
            batch_duration = batch_end - batch_start 
            batch_durations.append(batch_duration)
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        per_epoch_durations.append(epoch_duration)
    total_end = time.time()
    total_duration = total_end - total_start

print(f"Total duration is {total_duration}")
print(f"Per epoch duration {mean(per_epoch_durations)}")
print(f"Per batch duration {mean(batch_durations)}")