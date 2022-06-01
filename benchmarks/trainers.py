import time
import torch

def train(num_epochs, model, dl, per_epoch_durations, batch_durations, criterion, optimizer, p):
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
            # if i % 200 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.10f}')
            running_loss = 0.0

            batch_end = time.time()
            batch_duration = batch_end - batch_start 
            batch_durations.append(batch_duration)
            p.step()

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        per_epoch_durations.append(epoch_duration)