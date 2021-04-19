import torch.nn as nn
import torch
import numpy as np
from pathlib import Path
import random
import os
import dataloader
import model
import traintest
from stopping_criterion import StoppingCriterion 
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import roc
seed = 2
setting = {"layers": ["layer4"],                               "init": "normal",         "lr": 0.01, "mom": 0.01}
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
k = 10
EPOCHS = 75
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_acc_from_conf(conf_mat):
    r = conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]
    w = conf_mat[0][1] + conf_mat[0][2] + conf_mat[1][0] + conf_mat[1][2] + conf_mat[2][0] + conf_mat[2][1]
    return r / (w + r) 

def normalize(conf_mat):
    new_mat = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        total = 0
        for j in range(3):
            total += conf_mat[i][j]
        for j in range(3):
            new_mat[i][j] = conf_mat[i][j] / total
    return new_mat

def accs_from_confmat(confmat):
    normalacc = confmat[0][0] / (confmat[0][0] + confmat[0][1] + confmat[0][2])
    npacc = confmat[1][1] / (confmat[1][0] + confmat[1][1] + confmat[1][2])
    ipacc = confmat[2][2] / (confmat[2][0] + confmat[2][1] + confmat[2][2])

    return normalacc, ipacc, npacc

def main():
    dls = dataloader.get_fold_loaders(k, BATCH_SIZE)
    for i, d in enumerate(dls):
        torch.save(d, "loaders/" + str(i) + ".pth")

    accs = []
    norm_confmats = []
    confmats = []

    best_state_dict_init = torch.load("models/inits/best.pth")
    for fold in range(k):
        mod = model.get_pretrained_model(layer_names=setting["layers"], type_init=setting["init"]).to(device)
        mod.load_state_dict(best_state_dict_init)
        optim = model.get_optimizer(mod, feature_extract=True, lr=setting["lr"], mom=setting["mom"])
        criterion = nn.CrossEntropyLoss()
        for e in range(EPOCHS):
            mod, valloss, _, confmat = traintest.trainepoch(mod, dls[fold], criterion, optim, device)
            valacc = get_acc_from_conf(confmat)

            if e == EPOCHS-1: 
                confmats.append(confmat)
                norm_confmat = normalize(confmat)
                norm_confmats.append(norm_confmat)
                accs.append(valacc)
                torch.save(mod.state_dict(), "models/folds/" + str(fold))


main()

