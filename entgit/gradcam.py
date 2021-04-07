import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dataloader
import model
import os
import random

#Adatped from https://github.com/ramprs/grad-cam
device = torch.device("cpu")

# ResNet Class
class ResNet(nn.Module):
    def __init__(self, id):
        super(ResNet, self).__init__()
        # define the resnet152
        resnet = model.get_model()
        h = torch.load("models/" + str(id), map_location='cpu')

        #comment out if gpu vs cpu
        for key in list(h.keys()):
            h[key[7:]] = h.pop(key)
        resnet.load_state_dict(h)
        resnet.eval()
        self.resnet = resnet

        # isolate the feature blocks
        self.features = nn.Sequential(self.resnet.conv1,
                                      self.resnet.bn1,
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                                      self.resnet.layer1, 
                                      self.resnet.layer2, 
                                      self.resnet.layer3, 
                                      self.resnet.layer4)
        # average pooling layer
        self.avgpool = self.resnet.avgpool
        # classifier
        self.classifier = self.resnet.fc
        # gradient placeholder
        self.gradient = None

    # hook for the gradients
    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def forward(self, x):

        # extract the features
        x = self.features(x)

        # register the hook
        h = x.register_hook(self.activations_hook)

        # complete the forward pass
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)

        return x



def gradcam(idn):
    loaders = torch.load("loaders/" + str(idn) + ".pth")
    test_loader = loaders["val"]
    # init the resnet
    resnet = ResNet(idn)

    # set the evaluation mode
    resnet.eval()

    nc = 0
    ni = 0

    for img, label in test_loader:

        img, label = img.to(device), label.to(device)
        name = test_loader.dataset.last_name
        # forward pass
        out = resnet(img)

        pred = out.argmax(dim=1).numpy()[0]
        # get the gradient of the output with respect to the parameters of the model
        out[:, pred].backward()

        # pull the gradients out of the model
        gradients = resnet.get_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = resnet.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())
        #plt.show()

        # make the heatmap to be a numpy array
        heatmap = heatmap.numpy()
        to_name = name[name.rfind('\\') + 1:]
        correct = False

        if torch.argmax(out) == label:
            correct = True
        else:
            to_name = str(label.data.numpy()[0]) + str(torch.argmax(out).numpy()) + to_name

        # interpolate the heatmap
        path = './gradcam/used/' + name[name.rfind('\\', 0)+1:]
        img = cv2.imread(path)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        if correct:
            to_name = 'correct/' + to_name
        else:
            to_name = 'incorrect/' + to_name

        '''
        if idn == blank:
            print(to_name)
            cv2.imwrite('./gradcam/' + to_name, superimposed_img)
        '''

for i in range(10):
    gradcam(i)
