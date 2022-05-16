import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from metrics import HSIC, MinusRbfHSIC


def getActivation(name,hookLayersActivationDict):
  # the hook signature
  def hook(model, input, output):
    hookLayersActivationDict[name] = output.detach()
  return hook

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate(model,device,dataloader,hookLayerList,hookLayersActivationDict):
    """
    Evaluate the model using validation data and collect the activation data from each layers
    """
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    activationDict = {}
    for item in hookLayerList:
        activationDict[item] = []

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            target = target.to(device)
            data = data.to(device)
            
            # compute output
            output = model(data)
            output = output.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            top1.update(prec1.item(), data.size(0))
            for i in hookLayerList:
                activationDict[i].append(hookLayersActivationDict[i])
            break

    return top1.avg,activationDict

def getLayerWiseOutputCorrelation(hookLayersM1,hookLayersM2,activationDictM1,activationDictM2):
    """
    Compute CKA scores for every pair of layers from its corresponding activation data
    """
    col1 = []
    col2 = []
    hsicScoreList = []
    hsicObj = MinusRbfHSIC(sigma_x=1)
    for layer1 in hookLayersM1:
        for layer2 in hookLayersM2:
            #oaL1 = torch.flatten(activationDictM1[layer1][0])#activationDictM1[layer1][0].reshape(activationDictM1[layer1][0].size(0),-1)
            #oaL2 = torch.flatten(activationDictM2[layer2][0])#activationDictM2[layer2][0].reshape(activationDictM2[layer2][0].size(0),-1)
            oaL1 = activationDictM1[layer1][0].reshape(activationDictM1[layer1][0].size(0),-1)
            oaL2 = activationDictM2[layer2][0].reshape(activationDictM2[layer2][0].size(0),-1)
            hsicCross = HSIC(oaL1,oaL2).detach().item()
            hsicL1 = HSIC(oaL1,oaL1).detach().item()
            hsicL2 = HSIC(oaL2,oaL2).detach().item()
            # hsicCross = hsicObj(oaL1,oaL2).detach().item()
            # hsicL1 = hsicObj(oaL1,oaL1).detach().item()
            # hsicL2 = hsicObj(oaL2,oaL2).detach().item()
            denom=np.sqrt(hsicL1*hsicL2)
            if np.isnan(hsicL1) or np.isnan(hsicL2) or np.isnan(hsicCross) or denom == 0:
                print("Layer 1:"+layer1+", HSIC score:"+str(hsicL1))
                print("Layer 1:"+layer2+", HSIC score:"+str(hsicL2))
                print("Cross HSIC score:"+str(hsicCross))
                print("Denom:"+str(denom))
                continue
            finalScore = hsicCross/denom
            col1.append(hookLayersM1.index(layer1))
            col2.append(hookLayersM2.index(layer2))
            hsicScoreList.append(finalScore)
    return col1,col2,hsicScoreList