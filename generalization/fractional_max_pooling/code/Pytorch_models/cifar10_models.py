#!/usr/bin/env python
# coding: utf-8

# # Model for testing Generalizability of Deep Learning Systems
# ###### Models are constructed using pytorch

# In[1]:


# Import Necessary Packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np


# ## Fully Connected Network

# In[2]:


class FullyConnectedNet(nn.Module):
    ''' Fully Connected Network in Pytorch'''
    def __init__(self, hidden_neurons_, classes_, dropout_rate_, input_size_):
        '''
            hidden_neurons_ : list containing hidden neurons per layer
            classes_ : Number of classes of objects that will be classified
            droput_rate_ : gives the droput rate per network
            input_size_ : size of 1D array
        '''
        
        super(FullyConnectedNet, self).__init__()
        self.hn_ = hidden_neurons_
        self.c_ = classes_
        self.rate_ = dropout_rate_
        self.input_size_ = input_size_
        self.input = nn.Flatten()
        self.first = nn.Linear(input_size_, self.hn_[0])
        self.hidden_layers = nn.ModuleList(                                           [nn.Linear(self.hn_[idx -1], self.hn_[idx])                                            for idx in range(1, len(self.hn_))])
        self.dp = nn.ModuleList([nn.Dropout(p=self.rate_) for idx in range(len(self.hn_)+1)])
        self.last = nn.Linear(self.hn_[-1], self.c_)
        self.act = nn.ReLU()
        if self.c_ > 2:
            self.final_act = nn.Softmax(dim=0)
        else:
            self.final_act = nn.Sigmoid()
        
    def forward(self, X):
        
        out = X
        out = self.input(out)
        out = self.first(out)
        out = self.act(out)
        out = self.dp[0](out)
        for idx, layers in enumerate(self.hidden_layers):
            out = layers(out)
            out = self.act(out)
            out = self.dp[idx+1](out)
        out = self.last(out)
        out = self.final_act(out)
        return out


# ## Summary of Fully Connected Network

# In[3]:


hn_ = [256, 128]
c_ = 10
model = FullyConnectedNet(hn_, c_, 0.5, 3072)
summary(model, input_size=(3, 32, 32))


# # VGG Network - Pytorch

# ### Conv Layer

# In[4]:


class ConvLayer(nn.Module):
    ''' Convolutional Layer for VGG Network | Pytorch '''
    def __init__(self, in_channels, out_channels, kernel_size=3, k=0):
        '''
            in_channels = number of Channels in input
            out_channels = number of channels in output
            kernel_size = size of conv kernel
            k = dropout rate 
        '''

        super(ConvLayer, self).__init__()
        self.drop = False
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        if k > 0:
            self.drop = True
            self.dp = nn.Dropout(p=k)

    def forward(self, X):
        out = X
        out = self.conv_layer(out)
        if self.drop:
            out = self.dp(out)
        return out 


# ### Convolutional Block

# In[5]:


class ConvBlock(nn.Module):
    ''' Convolution Block for VGG Net'''
    def __init__(self, in_channels, out_channels, layers, kernel_size=3, k=0.5):
        '''
            in_channels = number of Channels in input
            out_channels = number of channels in output
            layers = number of convlayers
            kernel_size = size of conv kernel
            k = dropout rate 
        '''
        super(ConvBlock, self).__init__()
        self.m = False
        self.first = ConvLayer(in_channels=in_channels, out_channels=out_channels,             kernel_size=kernel_size, k=k)
        self.last = ConvLayer(in_channels=out_channels, out_channels=out_channels,             kernel_size=kernel_size, k=0)

        if layers > 2:
            self.m = True
            self.middle = nn.Sequential(
                *[ConvLayer(out_channels, out_channels, kernel_size, k=k) for _ in range(layers - 2)]
            )

    def forward(self, X):

        out = X
        out = self.first(out)
        if self.m:
            out = self.middle(out)
        out = self.last(out) 

        return out
        

        
        


# ### VGG Network using ConvBlock and Fully Connected Network

# In[6]:


class VGG(nn.Module):
    ''' VGG Net in Pytorch'''
    def __init__(self, in_channels, out_channel_list,layers,  h_list, classes, dp_list):
        '''
            in_channels = Number of Channels in Input
            out_channel_list = List containing number of channels in each convBlock
            h_list = list containing hidden neurons in Fully Connected Network
            layers = Number of layers in each Conv Layers
            classes = Number of Classes in CLassification
            dp_list = dropout list fot conv and Fully COnnected Layers    
        '''
        super(VGG, self).__init__()
        self.first = ConvBlock(in_channels=in_channels, out_channels=out_channel_list[0],layers=layers[0], k=dp_list[0])
        self.layers = nn.ModuleList(
            [ConvBlock(in_channels=out_channel_list[idx], \
                out_channels=out_channel_list[idx+1],\
                    layers=layers[idx+1], k=dp_list[0]) for idx in range(len(out_channel_list) - 1)]
        )
        self.max_pool = nn.ModuleList([nn.MaxPool2d(2) for idx in range(len(out_channel_list))])

        self.linear = FullyConnectedNet(h_list, classes,  dropout_rate_=dp_list[1], input_size_=out_channel_list[-1])

    def forward(self, X):
        out = X 
        out = self.first(out)
        out = self.max_pool[0](out)
        for idx , layer in enumerate(self.layers):
            out = layer(out)
            out = self.max_pool[idx+1](out)
        out = self.linear(out)
        
        return out


# ### VGG16 Summary

# In[7]:


k_list = [64, 128, 256, 512, 512]
l_list = [2, 2, 3, 3, 3]
h_list = [256, 128]
classes = 10 
dp_list = [0.4, 0.5]

model = VGG(in_channels=3, out_channel_list=k_list, layers=l_list, h_list=h_list, classes=classes, dp_list=dp_list)
summary(model, input_size=(3, 32, 32))


# # Fractional Max Pooling Network - Pytorch

# ### C2 Block

# In[8]:


class C2(nn.Module):
    ''' Conv Layer for Fractional Max Pooling Net '''
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(C2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, X):
        out = X
        out = self.conv_layer(out)
        return out


# ### C2 - FMP Layer

# In[9]:


class ConvFMPBlock(nn.Module):
    ''' C2- FMP Layer'''
    def __init__(self, in_channels, out_channels, fraction, kernel_size=3,):
        super(ConvFMPBlock, self).__init__()

        self.module = nn.Sequential(
            C2(in_channels=in_channels, out_channels=out_channels),
            nn.FractionalMaxPool2d(kernel_size=2, output_ratio=(fraction, fraction))
        )

    def forward(self, X):
        out = X
        out = self.module(out)

        return out


# In[12]:


### FMP Net 


# In[13]:


class FMPNet(nn.Module):
    ''' Fractional Max pooling VGG Like Network'''
    def __init__(self, in_channels, out_channel_list, dp_list, fraction, classes):
        super(FMPNet, self).__init__()
        self.layers = len(out_channel_list)
        self.first_layer = ConvFMPBlock(in_channels, out_channel_list[0], fraction=fraction)

        self.fmp_layers = nn.ModuleList(
            [ConvFMPBlock(out_channel_list[idx], out_channel_list[idx+1], fraction=fraction) for idx in range(self.layers - 1)]
        )

        self.dp = nn.ModuleList(
            [nn.Dropout(p=dp_list[idx]) for idx in range(self.layers)]
        )

        if classes > 2:
            self.final_act = nn.Softmax(dim=0)
        else:
            self.final_act = nn.Sigmoid()

        self.final = nn.Sequential(
            C2(out_channel_list[-1], out_channel_list[-1]),
            nn.Conv2d(out_channel_list[-1], out_channel_list[-1], kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(out_channel_list[-1]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(out_channel_list[-1], classes),
            self.final_act
        )
    def forward(self, X):

        out = X
        out = self.first_layer(out)
        out = self.dp[0](out)
        for idx, layers in enumerate(self.fmp_layers):
            out = layers(out)
            out = self.dp[idx+1](out)
        out = self.final(out)
        return out


# ### FMP-Net Summary

# In[14]:


k_list = [32, 64, 96, 128, 160, 192]
d_list = [0.3, 0.33, 0.37, 0.4, 0.45, 0.5]
Classes = 10
fraction = 1/np.sqrt(2)
m = FMPNet(3, k_list, d_list, fraction,Classes)
summary(m, input_size=(3, 32, 32))


# ### CompareNet

# In[15]:


class C2Maxpool2D(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(C2Maxpool2D, self).__init__()
        self.module = nn.Sequential(
            C2(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, X):
        out = X
        out = self.module(out)

        return out


# In[26]:


class MaxPool2DNet(nn.Module):

    def __init__(self, in_channels, out_channel_list, dp_list, classes):

        super(MaxPool2DNet, self).__init__()

        layer_length = len(out_channel_list)

        self.first = C2Maxpool2D(in_channels, out_channel_list[0])

        self.layers = nn.ModuleList(
            [C2Maxpool2D(out_channel_list[idx], out_channel_list[idx+1]) for idx in range(layer_length - 2)]
        )

        self.last_C2 = C2(out_channel_list[-2], out_channel_list[-1])

        self.dp = nn.ModuleList(
            [nn.Dropout(p=dp_list[idx]) for idx in range(layer_length)]
        )

        if classes > 2:
            self.final_act = nn.Softmax(dim=0)
        else:
            self.final_act = nn.Sigmoid()

        self.final = nn.Sequential(
            C2(out_channel_list[-1], out_channel_list[-1]),
            nn.Conv2d(out_channel_list[-1], out_channel_list[-1], kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(out_channel_list[-1]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(out_channel_list[-1], classes),
            self.final_act
        )

    def forward(self, X):

        out = X
        out = self.first(out)
        out = self.dp[0](out)
        for idx , layer in enumerate(self.layers):
            out = layer(out)
            out = self.dp[idx+1](out)

        out = self.last_C2(out)
        out = self.dp[-1](out)

        out = self.final(out)

        return out

        


# In[28]:


k_list = [32, 64, 96, 128, 160, 192]
d_list = [0.3, 0.33, 0.37, 0.4, 0.45, 0.5]
Classes = 10
m = MaxPool2DNet(3, k_list, d_list, Classes)
summary(m, input_size=(3, 32, 32))


# In[ ]:




