import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


def set_parameter_requires_grad(model, layer_names, feature_extracting=True):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if name[0:6] in layer_names:
            param.requires_grad = True


def get_pretrained_model(num_classes=3, layer_names=[], type_init = 'xavier_uniform'):
    pretrained_model = models.resnet152(pretrained=True)
    set_parameter_requires_grad(pretrained_model, layer_names)
    in_feats = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(
        in_feats, num_classes)
    if type_init == 'xavier_uniform':
        torch.nn.init.xavier_uniform_(pretrained_model.fc.weight)
    elif type_init == 'xavier_normal':
        torch.nn.init.xavier_normal_(pretrained_model.fc.weight)
    elif type_init =='uniform':
        torch.nn.init.uniform_(pretrained_model.fc.weight)
    elif type_init =='normal':
        torch.nn.init.normal_(pretrained_model.fc.weight)
    else:
        print(type_init, ' not recognized')


    return pretrained_model


def get_optimizer(model, feature_extract, lr, mom):

    params_to_update = model.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    # else:
        # for name, param in model.named_parameters():
            # if param.requires_grad == True:
                #print("\t", name)

    return optim.SGD(params_to_update, lr=lr, momentum=mom)


def get_params(model):
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
    return params_to_update


def get_model(num_classes=3):
    pretrained_model = models.resnet152(pretrained=True)
    in_feats = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(
        in_feats, num_classes)

    return pretrained_model
