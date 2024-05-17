import torch
import torch.nn as nn

def train(dataloader, model, loss_fn, optimizer, device, print_interval=100):
    size = len(dataloader.dataset)
    model.train()
    batch_size = dataloader.batch_size
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_interval == 0:
            loss, current = loss.item(), batch * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")