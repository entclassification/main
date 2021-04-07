import torch
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_confusion_matrix(true, pred):

    K = len(np.unique(true))
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def calc_accuracy(out, labels, find_indices=True):
    num_correct = 0
    total = 0
    preds = []
    trues = []
    for o, labs in zip(out, labels):
        if find_indices:
            _, max_indices = torch.max(o, 1)
        else:
            max_indices = o
        num_correct += (max_indices == labs).sum().item()
        total += max_indices.size()[0]
        for p, l in zip(max_indices, labs):
            preds.append(p)
            trues.append(l)

    return float(num_correct) / float(total), preds, trues


def test(mod, dataloader, criterion):

    val_losses = []
    outs = []
    labels = []
    mod.eval()
    dataloader.dataset.test_mode()
    for img_batch, label in dataloader:

        img_batch = img_batch.to(device)
        label = label.to(device)

        with torch.set_grad_enabled(False):
            out = mod(img_batch)
            loss = criterion(out, label.long())

        val_losses.append(loss.item())
        outs.append(out)
        labels.append(label)

    acc, preds, trues = calc_accuracy(outs, labels)

    mat = compute_confusion_matrix(trues, preds)
    return acc, mat

def trainepoch(mod, dataloaders, criterion, optimizer, device, validate=True):

    mod.train()
    dataloaders["train"].dataset.train_mode()
    for img_batch, labels in dataloaders["train"]:
        # zero the parameter gradients
        img_batch = img_batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Get model outputs and calculate loss
            outs = mod(img_batch)
            loss = criterion(outs, labels.long())
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

        # statistics
    val_losses = []
    outs = []
    labels = []
    mod.eval()
    if validate:
        dataloaders["val"].dataset.test_mode()
        for img_batch, label in dataloaders["val"]:
            img_batch = img_batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                out = mod(img_batch)
                loss = criterion(out, label.long())
                # backward + optimize only if in training phase
            val_losses.append(loss.item())
            outs.append(out)
            labels.append(label)
        acc, preds, trues = calc_accuracy(outs, labels)
        conf_mat = compute_confusion_matrix(trues, preds)
        return mod, np.mean(val_losses), acc, conf_mat
    else:
        return mod, None, None, None



def fulltrain(mod, config, dataloaders, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        mod, _, _, _ = trainepoch(mod, dataloaders, criterion, optimizer, device)
    test(mod, dataloaders["test"])

    torch.save(mod.state_dict(), "model")










