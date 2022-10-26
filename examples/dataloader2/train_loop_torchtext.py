# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchtext
import torchtext.functional as F

import torchtext.transforms as T
from torch.hub import load_state_dict_from_url
from torch.optim import AdamW
from torchdata.dataloader2 import DataLoader2
from torchtext.datasets import SST2


LEARNING_RATE = 1e-5
PADDING_IDX = 1
BOS_IDX = 0
EOS_IDX = 2
MAX_SEQ_LEN = 256


XLMR_VOCAB_PATH = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
XLMR_SPM_MODEL_PATH = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

text_transform = T.Sequential(
    T.SentencePieceTokenizer(XLMR_SPM_MODEL_PATH),
    T.VocabTransform(load_state_dict_from_url(XLMR_VOCAB_PATH)),
    T.Truncate(MAX_SEQ_LEN - 2),
    T.AddToken(token=BOS_IDX, begin=True),
    T.AddToken(token=EOS_IDX, begin=False),
)

NUM_EPOCHS = 1
BATCH_SIZE = 8
NUM_CLASSES = 2
INPUT_DIM = 768


def apply_transform(x):
    return text_transform(x[0]), x[1]


def train_step(input: torch.Tensor, target: torch.Tensor) -> None:
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()


def eval_step(input: torch.Tensor, target: torch.Tensor) -> None:
    output = model(input)
    loss = criteria(output, target).item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()


def evaluate() -> None:
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            input = F.to_tensor(batch["token_ids"], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            loss, predictions = eval_step(input, target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1

    return total_loss / counter, correct_predictions / total_predictions


if __name__ == "__main__":

    train_datapipe = SST2(split="train")
    eval_datapipe = SST2(split="dev")

    train_datapipe = train_datapipe.map(apply_transform)
    train_datapipe = train_datapipe.batch(BATCH_SIZE)
    train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
    train_dataloader = DataLoader2(datapipe=train_datapipe)
    print("Created train dataloader")

    eval_datapipe = eval_datapipe.map(apply_transform)
    eval_datapipe = eval_datapipe.batch(BATCH_SIZE)
    eval_datapipe = eval_datapipe.rows2columnar(["token_ids", "target"])
    eval_dataloader = DataLoader2(datapipe=eval_datapipe)
    print("Created eval dataloader")

    classifier_head = torchtext.models.RobertaClassificationHead(num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
    model = torchtext.models.XLMR_BASE_ENCODER.get_model(head=classifier_head)
    model.to(DEVICE)

    optim = AdamW(model.parameters(), lr=LEARNING_RATE)
    criteria = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(train_dataloader):
            input = F.to_tensor(batch["token_ids"], padding_value=PADDING_IDX).to(DEVICE)
            target = torch.tensor(batch["target"]).to(DEVICE)
            train_step(input, target)

            # stop early for example purpose
            if step == 10:
                break

        loss, accuracy = evaluate()
        print(f"Epoch: {epoch}, loss: {loss}, accuracy: {accuracy}")

    print("Finished Training")
