import torch
import logging
import os
import pickle
import sys

import modules.models as models
import modules.models_tjr1 as models_tjr1
import modules.models_tjr2 as models_tjr2

from dataloader import tfloader


def getModel(model: str, opt, dataset=None, source=None):
    model = model.lower()
    if dataset == 'coat' and source == 'unif':
        opt['use_bn'] = True
    else:
        opt['use_bn'] = False

    if model == "fm":
        return models.FM(opt)
    elif model == "dnn":
        return models.DNN(opt)
    elif model == "deepfm":
        return models.DeepFM(opt)
    elif model == "dcn":
        return models.DeepCrossNet(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def getModel_TJR1(model: str, opt):
    model = model.lower()
    if model == "fm":
        return models_tjr1.FM(opt)
    elif model == "dnn":
        return models_tjr1.DNN(opt)
    elif model == "deepfm":
        return models_tjr1.DeepFM(opt)
    elif model == "dcn":
        return models_tjr1.DeepCrossNet(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def getModel_TJR2(model: str, opt):
    model = model.lower()
    if model == "fm":
        return models_tjr2.FM(opt)
    elif model == "dnn":
        return models_tjr2.DNN(opt)
    elif model == "deepfm":
        return models_tjr2.DeepFM(opt)
    elif model == "dcn":
        return models_tjr2.DeepCrossNet(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def getOptim(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))


def getDevice(device_id):
    # print(device_id)
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        # torch.cuda.set_device(device_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def getDataLoader(dataset: str, path):
    dataset = dataset.lower()
    if dataset == 'coat':
        return tfloader.CoatLoader(path)
    elif dataset == 'kuairand':
        return tfloader.KuaiRandLoader(path)


def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    return [i + 1 for i in list(defaults.values())]


def get_log(name=""):
    FORMATTER = logging.Formatter(fmt="[{asctime}]:{message}", style='{')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    return logger
