import torch
from tqdm import tqdm
from torch import nn
from tqdm import tqdm
import numpy as np

#import wandb

def train(model, train_loader, val_loader, optimizer, scheduler, epochs, device, use_wandb=True):

    train_losses, val_losses = [], []
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(epochs)):

        train_epoch_losses = []

        model.train()
        for batch in train_loader:

            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)
            loss = loss_fn(output, batch.y)
            loss.backward()
            optimizer.step()

            train_epoch_losses.append( loss.item() )

        train_losses.append(np.mean(train_epoch_losses))

        val_epoch_losses = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                output = model(batch)
                loss = loss_fn(output, batch.y)

                val_epoch_losses.append( loss.item() )

        val_losses.append(np.mean(val_epoch_losses))

        scheduler.step()

        # if use_wandb:
        #     wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr})

    return train_losses, val_losses