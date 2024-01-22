# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch
import torch.distributed as dist
from torch import nn

from torchdata.dataloader2 import DataLoader2, DistributedReadingService
from torchdata.datapipes.iter import IterableWrapper


class ToyModel(nn.Module):
    def __init__(self) -> None:
        """
        In the model constructor, we instantiate four parameters and use them
        as member parameters.
        """
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))
        self.b = nn.Parameter(torch.randn(()))
        self.c = nn.Parameter(torch.randn(()))
        self.d = nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple model forward function
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3


def main() -> None:
    model = ToyModel()

    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(2)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "0"

    dist.init_process_group("gloo")

    # Use a prime number to make sure uneven data sharding and let
    # DistributedReadingService prevent hanging with the unbalanced data shard
    data_length = 19997

    train_features = IterableWrapper([torch.rand(3) for _ in range(data_length)])
    train_labels = IterableWrapper([torch.rand(3) for _ in range(data_length)])

    # sharding_filter will automatically shard the data based on the
    # distributed ranks
    train_data_pipe = train_features.zip(train_labels).shuffle().sharding_filter()

    # Torch Distributed is required to use DistributedReadingService
    reading_service = DistributedReadingService()

    # Create DataLoader2 with DistributedReadingService
    data_loader2 = DataLoader2(
        datapipe=train_data_pipe,
        reading_service=reading_service,
    )

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    for epoch in range(5):

        # Set manual seed per epoch to control the randomness for shuffle.
        torch.manual_seed(epoch)

        running_loss = 0.0
        for step, data in enumerate(data_loader2):
            train_feature, train_label = data
            optimizer.zero_grad()

            predicted_outputs = model(train_feature)
            loss = criterion(predicted_outputs, train_label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 2000 == 1999:
                print("[epoch: %d, %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0

    print("Finished Training")

    """
    Training Output:

    [epoch: 1,  2000] loss: 0.860
    [epoch: 1,  4000] loss: 0.823
    [epoch: 1,  6000] loss: 0.809
    [epoch: 1,  8000] loss: 0.778
    [epoch: 1, 10000] loss: 0.753
    [epoch: 1, 12000] loss: 0.756
    [epoch: 1, 14000] loss: 0.730
    [epoch: 1, 16000] loss: 0.727
    [epoch: 1, 18000] loss: 0.704
    [epoch: 1, 20000] loss: 0.703
    [epoch: 2,  2000] loss: 0.677
    [epoch: 2,  4000] loss: 0.649
    [epoch: 2,  6000] loss: 0.648
    [epoch: 2,  8000] loss: 0.629
    [epoch: 2, 10000] loss: 0.623
    [epoch: 2, 12000] loss: 0.593
    [epoch: 2, 14000] loss: 0.586
    [epoch: 2, 16000] loss: 0.584
    [epoch: 2, 18000] loss: 0.571
    [epoch: 2, 20000] loss: 0.558
    [epoch: 3,  2000] loss: 0.537
    [epoch: 3,  4000] loss: 0.540
    [epoch: 3,  6000] loss: 0.544
    [epoch: 3,  8000] loss: 0.512
    [epoch: 3, 10000] loss: 0.496
    [epoch: 3, 12000] loss: 0.506
    [epoch: 3, 14000] loss: 0.486
    [epoch: 3, 16000] loss: 0.489
    [epoch: 3, 18000] loss: 0.489
    [epoch: 3, 20000] loss: 0.456
    [epoch: 4,  2000] loss: 0.474
    [epoch: 4,  4000] loss: 0.445
    [epoch: 4,  6000] loss: 0.442
    [epoch: 4,  8000] loss: 0.440
    [epoch: 4, 10000] loss: 0.434
    [epoch: 4, 12000] loss: 0.421
    [epoch: 4, 14000] loss: 0.415
    [epoch: 4, 16000] loss: 0.404
    [epoch: 4, 18000] loss: 0.427
    [epoch: 4, 20000] loss: 0.410
    [epoch: 5,  2000] loss: 0.395
    [epoch: 5,  4000] loss: 0.393
    [epoch: 5,  6000] loss: 0.389
    [epoch: 5,  8000] loss: 0.397
    [epoch: 5, 10000] loss: 0.375
    [epoch: 5, 12000] loss: 0.375
    [epoch: 5, 14000] loss: 0.372
    [epoch: 5, 16000] loss: 0.365
    [epoch: 5, 18000] loss: 0.371
    [epoch: 5, 20000] loss: 0.359
    Finished Training

    """


if __name__ == "__main__":
    main()  # pragma: no cover
