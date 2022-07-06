# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import IterableWrapper


class ToyModel(torch.nn.Module):
    def __init__(self):
        """
        In the model constructor, we instantiate four parameters and use them
        as member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        Simple model forward function
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3


if __name__ == "__main__":
    model = ToyModel()

    train_features = IterableWrapper([torch.rand(3) for _ in range(20000)])
    train_labels = IterableWrapper([torch.rand(3) for _ in range(20000)])
    train_data_pipe = train_features.zip(train_labels).shuffle()

    # DataLoader2 wraps an iterable around the Datapipe to enable easy access to
    # the features and labels.
    data_loader = DataLoader2(datapipe=train_data_pipe)

    # Construct the loss function and the optimizer.
    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    # Loop over the dataset multiple times. Here we are doing only 3 training
    # epochs - that is, three passes over the training datapipes.
    for epoch in range(3):

        # Set manual seed per epoch to control the randomness for shuffle.
        torch.manual_seed(epoch)

        running_loss = 0.0
        for step, data in enumerate(data_loader):
            # Obtain the inputs and labels from data.
            train_feature, train_label = data

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Train step: forward + backward + optimize.
            predicted_outputs = model(train_feature)
            loss = criterion(predicted_outputs, train_label)
            loss.backward()
            optimizer.step()

            # Calculate the statistics.
            running_loss += loss.item()
            # Print the loss every 2000 mini-batches.
            if step % 2000 == 1999:
                print("[epoch: %d, %5d] loss: %.3f" % (epoch + 1, step + 1, running_loss / 2000))
                running_loss = 0.0

    print("Finished Training")
