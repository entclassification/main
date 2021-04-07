import torch
import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from numpy import *
from PIL import Image
import torchvision.transforms as transforms
from copy import copy

IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class dataset_(Dataset):

    def __init__(self, mode):
        self.mode = mode
        self.paths, self.labels = self.get_paths_and_labels()
        self.paths = np.array(self.paths)
        self.labels = np.array(self.labels)
        self.transform_n = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_mode()
        self.last_name = ""

    def get_paths_and_labels(self):
        normal_paths = sorted(
            glob.glob("Data/All/Normal/*"))
        ip_paths = sorted(glob.glob("Data/All/IP/*"))
        np_paths = sorted(glob.glob("Data/All/NP/*"))
        norm_label = 0
        ip_label = 1
        np_label = 2
        paths = []
        labels = []

        for path in normal_paths:
            paths.append(path)
            labels.append(norm_label)
        for path in ip_paths:
            paths.append(path)
            labels.append(ip_label)
        for path in np_paths:
            paths.append(path)
            labels.append(np_label)
        return paths, labels

    def test_mode(self):
        self.transform_c = transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE))

    def train_mode(self):
        rand_crop = transforms.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE), scale=(0.50, 0.60))
        self.transform_c = transforms.Compose([rand_crop])

    def __len__(self):

        length = len(self.labels)
        return length

    def __getitem__(self, index):
        p = self.paths[index]
        self.last_name = p[p.rfind('/', 0, p.rfind('/')):]
        x = Image.open(self.paths[index])
        if self.transform_c is not None:
            x = self.transform_c(x)
        na = p[p.rfind('\\', 0)+1:]
        x.save("gradcam/used/" + na)
        x = np.array(x)
        x = np.moveaxis(x, [2, 1], [0, 2])
        x = torch.from_numpy(x).float().to(device)
        x /= 255
        x = self.transform_n(x)

        label = self.labels[index]

        return x, label


def get_loaders(batch_size, split=1.0):
    # torch specified mean and std for pre trained net
    train_inds = []
    val_inds = []

    last = 0
    train_dataset = dataset_("train_val")

    for i in range(3):
        n = np.count_nonzero(train_dataset.labels == i)
        indices = list(range(last, last + n))
        last += n
        train_split = int(n * split)
        val_split = int(n * (1-split))
        np.random.shuffle(indices)
        train_inds.append(indices[:train_split])
        val_inds.append(indices[train_split:train_split + val_split])
    train_inds = np.concatenate(train_inds)
    val_inds = np.concatenate(val_inds)

    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)


    test_dataset = dataset_("test")

    traindataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valdataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler)

    testdataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)

    return {'train': traindataloader, 'val': valdataloader, 'test': testdataloader}


def get_fold_loaders(k, batch_size):

    traindataloaders = []
    valdataloaders = []
    gradcamloaders =[]
    train_split_sizes = []
    val_split_sizes = []

    train_dataset = dataset_("train_val")
    shuffled_indices = []
    last = 0
    for i in range(3):
        n = np.count_nonzero(train_dataset.labels == i)
        indices = list(range(last, last + n))
        last += n
        train_split_sizes.append(int(n * (k-1 / k)))
        val_split_sizes.append(int(n * (1 / k)))
        np.random.shuffle(indices)
        shuffled_indices.append(indices)
        

    for fold in range(k):
        train_inds = []
        val_inds = []
        for c in range(3):
            inds = shuffled_indices[c]
            val_split = inds[fold*val_split_sizes[c]:(fold+1)*val_split_sizes[c]]
            #print(val_split)
            train_split = inds.copy()
            for i in val_split:
                train_split.remove(i)
            train_inds.append(train_split)
            val_inds.append(val_split)
        train_inds = np.concatenate(train_inds)
        val_inds = np.concatenate(val_inds)
        train_sampler = SubsetRandomSampler(train_inds)
        val_sampler = SubsetRandomSampler(val_inds)
        traindataloaders.append(DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler))

        valdataloaders.append(DataLoader(train_dataset, batch_size=1, sampler=val_sampler))
    return [{"train": train, "val": val} for train, val in zip(traindataloaders, valdataloaders)]



