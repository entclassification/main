import matplotlib.pyplot as plt
import dataloader
import numpy as np
import torch.nn.functional as F
import model
import torch
import scikitplot as skplt
from numpy import array
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
import random
from b import binarize




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)
random.seed(7)

def get_net(id):

    resnet = model.get_model()
    h = torch.load("models/" + str(id) , map_location='cuda:0')
    resnet.load_state_dict(h)
    resnet.eval()
    resnet.to(device)
    return resnet


    return mat
def get_guesses(preds):
    guesses = []
    for p in preds:
        if 1 - p > 0.33333333:
            guesses.append(0)
        else:
            guesses.append(1)
    return guesses



def get_preds(net, loader):
    preds = []
    labels = []
    for img, label in loader:
        img = img.to(device)
        label = label.to(device)
        labels.append(label.item())
        preds.append(F.softmax(net(img)).cpu().view(3).data.numpy())
    return preds, labels

def get_acc(preds, labels):
    correct = 0
    total = len(labels)
    np.set_printoptions(precision=3, suppress=True)
    new_preds = []
    for p, la in zip(preds, labels):

        if np.argmax(p) == la:
            correct += 1

        new_preds.append(p)

    return new_preds, labels, correct / total



def main():
    tprs = [[],[],[]]
    fprs = [[],[],[]]
    aucs = [[], [], []]
    specs = [[], [], []]
    sens = [[], [], []]
    accs = []
    fones = [[], [], []]
    fbest = 0
    base_fpr = np.linspace(0, 1, 101)
    loaders = [torch.load("loaders/"+str(i)+".pth") for i in range(10)]
    for i in range(len(loaders)):
        net = get_net(i)
        preds, labels = get_preds(net, loaders[i]["val"])

        for c in range(3):
            scores, blabels = binarize(preds, labels, c)
            conf_mat = confusion_matrix(blabels, get_guesses(scores))

            tn, fp, fn, tp = conf_mat.ravel()
            sen = tp / (tp+fn)
            specificity = tn / (tn + fp)
            fone = tp / (tp + 0.5*(fp + fn)) 
            fones[c].append(fone)
            specs[c].append(specificity)
            sens[c].append(sen)
            fpr, tpr, _ = roc_curve(blabels, scores)
            aucs[c].append(auc(fpr, tpr))

            tpr = interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            tprs[c].append(tpr)
    c = 0
    np.save("tprs", tprs)
    '''
    fones = np.array(fones)
    specs = np.array(specs)
    sens = np.array(sens)
    print("FONES")
    for f in fones:
        print(np.mean(f))
        print(np.std(f))
    print("SPECS and SENS STD")
    for c in range(3):
        print("spec")
        print(np.mean(specs[c]))
        print(np.std(specs[c]))
        print("sens")
        print(np.mean(sens[c]))
        print(np.std(sens[c]))
    '''

main()