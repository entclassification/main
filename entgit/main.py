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

setting = {"layers": ["layer4"],                               "init": "normal",         "lr": 0.01, "mom": 0.01}
torch.manual_seed(9)
random.seed(9)

test_dataloader = dataloader.get_loaders(1)["test"]
stopcrit = StoppingCriterion(20, 5)
RUNS = 30
EPOCHS = 75
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def convert_to_name(setting):
    return str(setting["layers"]) + '_' + setting["init"] + "_" + str(setting["lr"]) + "_" + str(setting["mom"])

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)
def get_preds(net):
    preds = []
    labels = []
    net.eval()
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        labels.append(label.item())
        preds.append(F.softmax(net(img)).cpu().view(3).data.numpy())
    return preds, labels

def plot_run(name, n, history, folder):
    Path(folder + "/" + name + "/" + n).mkdir(parents=True, exist_ok=True)
    epoch_axis = np.arange(len(history["loss"]))
    plt.plot(epoch_axis, history["loss"])
    plt.title("valloss")
    plt.savefig(folder + "/" + name + "/" + n + "/valloss.png")
    plt.close()

    plt.plot(epoch_axis, history["acc"])
    plt.title("valacc")
    plt.savefig(folder + "/" + name + "/" + n + "/valacc.png")
    plt.close()

    plt.plot(epoch_axis, history["normacc"])
    plt.title("normalacc")
    plt.savefig(folder + "/" + name + "/" + n + "/normalacc.png")
    plt.close()

    plt.plot(epoch_axis, history["ipacc"])
    plt.title("ipacc")
    plt.savefig(folder + "/" + name + "/" + n + "/ipacc.png")
    plt.close()

    plt.plot(epoch_axis, history["npacc"])
    plt.title("npacc")
    plt.savefig(folder + "/" + name + "/" + n + "/npacc.png")
    plt.close()

def plot_avg_runs(name, histories, folder="figures"):
    accs = np.array([np.array(h["acc"]) for h in histories])
    mean_acc, std = tolerant_mean(accs)
    plt.plot(np.arange(len(mean_acc)) + 1, mean_acc)
    plt.fill_between(np.arange(len(mean_acc)) + 1, mean_acc - std, mean_acc + std)
    plt.title("meanvalacc")
    plt.savefig(folder + "/" + name + "/meanvalacc.png")
    plt.close()


def run(setting, n, save_dir, folder, early_stop=True, split=0.75, init_dict=None):
    name = convert_to_name(setting)
    model_save_dir = save_dir + '/'

    history = {"loss": [], "acc": [], "normacc": [], "ipacc": [], "npacc": [], "confmat": [], "best_avg": 0}

    mod = model.get_pretrained_model(layer_names=setting["layers"], type_init=setting["init"]).to(device)
    if init_dict is not None:
        mod.load_state_dict(init_dict)
    optim = model.get_optimizer(mod, feature_extract=True, lr=setting["lr"], mom=setting["mom"])
    criterion = nn.CrossEntropyLoss()
    Path(model_save_dir + name + "/" + n).mkdir(parents=True, exist_ok=True)
    torch.save(mod.state_dict(), model_save_dir + name + "/" + n + '/epoch_0')
    stop = False
    if early_stop:
        dataloaders = dataloader.get_loaders(BATCH_SIZE, split)
        while not stop:
            print(stopcrit.checks)
            mod, valloss, valacc, confmat = traintest.trainepoch(mod, dataloaders, criterion, optim, device)
            #normalacc, ipacc, npacc = accs_from_confmat(confmat)
            history["loss"].append(valloss)
            history["acc"].append(valacc)
            #history["normacc"].append(normalacc)
            #history["ipacc"].append(ipacc)
            #history["npacc"].append(npacc)
            history["confmat"].append(confmat)
            stop = stopcrit.check(valacc, mod.state_dict())
    else:
        dataloaders = dataloader.get_loaders(BATCH_SIZE, split)
        for epoch in range(EPOCHS):
            if split == 1.0:
                validate = False
            else:
                validate = True
            mod, valloss, valacc, confmat = traintest.trainepoch(mod, dataloaders, criterion, optim, device, validate)
            if valloss is not None:
                #normalacc, ipacc, npacc = accs_from_confmat(confmat)
                history["loss"].append(valloss)
                history["acc"].append(valacc)
                #history["normacc"].append(normalacc)
                #history["ipacc"].append(ipacc)
                #history["npacc"].append(npacc)
                history["confmat"].append(confmat)
                stop = stopcrit.check(valacc, mod.state_dict())

    if split != 1.0:
        history["best_avg"] = stopcrit.last_avg
        torch.save(stopcrit.best_model_dict, model_save_dir + name + "/" + n + '/epoch_' + str(stopcrit.best_check))
        plot_run(name, n, history, folder)
        best_acc = stopcrit.best_val
        best_epoch = stopcrit.best_check
        stopcrit.reset()
    else:
        torch.save(mod.state_dict(), model_save_dir + name + "/" + n + "/epoch_" + str(EPOCHS))
        best_acc = None
        best_epoch = None
    return history, best_acc, best_epoch



def main():
    #group_name = config.type_init + str(config.lr) + '-' + str(config.mom) 
    histories = []
    best_avgs = []
    name = convert_to_name(setting)
    for r in range(RUNS):
        history, _, _ = run(setting, str(r), 'models/dif_init',"figures/dif_init")
        histories.append(history)
        best_avgs.append(history["best_avg"])

    plot_avg_runs(name, histories, "figures/dif_init")
    best_ind = np.argmax(best_avgs)
    
    best_state_dict_init = torch.load("models/inits/" + name + "/" + str(best_ind) + '/epoch_0')
    torch.save(best_state_dict_init, "models/inits/best.pth")

    histories = []
    best_avgs = []

if __name__ == '__main__':
    main()
