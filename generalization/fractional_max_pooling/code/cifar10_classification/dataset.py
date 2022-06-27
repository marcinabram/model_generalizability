#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split


# In[2]:


def getDatset(data, val_size=1000):
    if data=="cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_val_set = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)


        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_eval)


        train_size = len(train_val_set) - val_size

        trainset, valset = random_split(train_val_set, [train_size, val_size])

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        valloader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False, num_workers=2)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=2)

        dataloader = [trainloader, valloader, testloader]

        return dataloader


# In[ ]:




