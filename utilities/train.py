from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os.path as osp
import copy
import sys
from tqdm import tqdm
from utils import AverageMeter,accuracy
from resnet import resnet18,resnet34,resnet50,lambda_resnet50


MODEL_NAME = sys.argv[1]
EPOCHS = int(sys.argv[2])
SEED = int(sys.argv[3])
MODEL_DUMP_PATH = osp.join(sys.argv[4],MODEL_NAME+"_seed_"+str(SEED)+"_epochs_"+str(EPOCHS)+".pth")
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
NUM_WORKERS = 4

def train(model,device,dataloader,optimizer):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    for _, (data, target) in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
        data,target = data.to(device),target.to(device)
        # compute output
        
        output = model(data)
        loss = criterion(output, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output, target)[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1, data.size(0))

    return losses.avg,top1.avg



def validate(model,device,dataloader):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(dataloader, desc="Iteration",file=sys.stdout)):
            target = target.to(device)
            data = data.to(device)
            

            # compute output
            output = model(data)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

    return losses.avg,top1.avg



torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
criterion = nn.CrossEntropyLoss()

normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])

train_dataset = datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_NAME == "resnet18":
    model = resnet18(num_classes=100)
elif MODEL_NAME == "resnet34":
    model = resnet34(num_classes=100)
elif MODEL_NAME == "resnet50":
    model = resnet50(num_classes=100)
elif MODEL_NAME == "lambda_resnet50":
    model = lambda_resnet50(num_classes=100)
else:
    print("Please correct the model name")
    exit(1)
    
model = model.to(device)
NUM_EPOCHS = EPOCHS

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=5e-4)
cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS)

for ep in range(NUM_EPOCHS):
    train_loss,train_acc = train(model,device,train_loader,optimizer)
    valid_loss,valid_acc = validate(model,device,val_loader)
    cosine_lr_scheduler.step()
    if (ep%10 == 0):
        print("Train loss:{}, Validation loss:{}".format(train_loss,valid_loss))
        print("Train acc:{}, Validation acc:{}".format(train_acc,valid_acc))
        
        
torch.save(model.state_dict(), MODEL_DUMP_PATH)
#print(model)
