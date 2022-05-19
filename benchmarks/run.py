import argparse
from torchvision.prototype.datasets import load
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="gtsrb", help="The name of the dataset")
parser.add_argument("--model_name", type=str, default="resnext50_32x4d", help="The name of the model")
parser.add_argument("--batch_size", type=int, default=32, help="")
parser.add_argument("--report_location", type=str, default="./report.md", help="The location where the generated report will be stored")

args = parser.parse_args()
dataset = args.dataset
batch_size = args.batch_size

# setup data pipe
dp = load("gtsrb", split="train")
print(f"batch size {batch_size}")
print(f"Dataset name {dp}")
print(f"Dataset length {len(dp)}")

# Setup data loader
# Shuffle won't work in distributed 
dl = DataLoader(dataset=dp, batch_size=batch_size, shuffle=True)

# Training loop
for elem in dl:
    print(i)