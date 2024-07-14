import os
import torch
import torch.nn as nn
import torch.distributions as td
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import matplotlib.pyplot as plt
from tqdm.notebook import trange 
import random
from torch.utils.data import DataLoader, TensorDataset
from models import RealNVP
from models import AutoEncoderA, AutoEncoderC, AutoEncoderA2, AutoencoderA3, AutoEncoderB
from sklearn.model_selection import train_test_split
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import gc
import pandas as pd
from torch import distributions





class AutoEncoder(nn.Module):
    """
    A convolutional autoencoder neural network for unsupervised learning of compressed representations of input data.

    Args:
        ksize (list): List of kernel sizes for each convolutional layer.
        knum (list): List of output channels for each convolutional layer.
        kstride (list): List of strides for each convolutional layer.
        Cdroprate (list): List of dropout rates for each convolutional layer.
        Ctdroprate (list): List of dropout rates for each transposed convolutional layer.
        Ddroprate (list): List of dropout rates for each dense layer.
        Dtdroprate (list): List of dropout rates for each transposed dense layer.
        z_drop (float): Dropout rate for the latent space.
        activ (nn.Module): Activation function to use (default is Tanh).
        padding (int): Padding for convolutional layers.
        input_shape (list): Shape of the input data (default is [1, 32, 32]).
        Dsize (list): List of layer sizes for the dense layers.
        z_dim (int): Dimension of the latent space.
        bias (bool): Whether to use biases in the layers (default is False).

    Attributes:
        activ (nn.Module): Activation function.
        input_shape (list): Shape of the input data.
        z_dim (int): Dimension of the latent space.
        z_drop (float): Dropout rate for the latent space.
        hidden (int): Number of channels in the deepest convolutional layer.
        CDP_layers (nn.ModuleList): List of convolutional layers.
        CDPt_layers (nn.ModuleList): List of transposed convolutional layers.
        Dlayers (nn.ModuleList): List of dense layers.
        Dtlayers (nn.ModuleList): List of transposed dense layers.
        out_dim (int): Flattened size of the output from the last convolutional layer.
        out_n_pix (int): Number of pixels in the output from the last convolutional layer.
        outD (nn.Linear): Linear layer mapping to the latent space.
        inD (nn.Linear): Linear layer mapping from the latent space.
        flatten (nn.Flatten): Flattening layer.

    Model Structure:
        - Encoder:
            - Convolutional layers with ReLU activation and dropout:
                - Conv2d(input channels, knum[0], kernel_size=ksize[0], stride=kstride[0], padding=padding)
                - Dropout(Cdroprate[0])
                - Activation(activ)
                - Conv2d(knum[0], knum[1], kernel_size=ksize[1], stride=kstride[1], padding=padding)
                - Dropout(Cdroprate[1])
                - Activation(activ)
                - ...
                - Conv2d(knum[-2], knum[-1], kernel_size=ksize[-1], stride=kstride[-1], padding=padding)
                - Dropout(Cdroprate[-1])
                - Activation(activ)
            - Flatten layer
            - Fully connected dense layers:
                - Linear(flattened size, Dsize[0])
                - Dropout(Ddroprate[0])
                - Activation(activ)
                - Linear(Dsize[0], Dsize[1])
                - Dropout(Ddroprate[1])
                - Activation(activ)
                - ...
                - Linear(Dsize[-2], Dsize[-1])
                - Dropout(Ddroprate[-1])
                - Activation(activ)
            - Latent space layer:
                - Linear(Dsize[-1], z_dim)

        - Decoder:
            - Fully connected dense layers:
                - Linear(z_dim, Dsize[-1])
                - Dropout(z_drop)
                - Activation(activ)
                - Linear(Dsize[-1], Dsize[-2])
                - Dropout(Dtdroprate[-1])
                - Activation(activ)
                - ...
                - Linear(Dsize[1], Dsize[0])
                - Dropout(Dtdroprate[1])
                - Activation(activ)
                - Linear(Dsize[0], flattened size)
            - Reshape layer to match the output of the last convolutional layer in the encoder
            - Transposed convolutional layers with ReLU activation and dropout:
                - ConvTranspose2d(knum[-1], knum[-2], kernel_size=ksize[-1], stride=kstride[-1], padding=padding, output_padding=1)
                - Dropout(Ctdroprate[-1])
                - Activation(activ)
                - ConvTranspose2d(knum[-2], knum[-3], kernel_size=ksize[-2], stride=kstride[-2], padding=padding, output_padding=1)
                - Dropout(Ctdroprate[-2])
                - Activation(activ)
                - ...
                - ConvTranspose2d(knum[1], knum[0], kernel_size=ksize[1], stride=kstride[1], padding=padding, output_padding=1)
                - Dropout(Ctdroprate[1])
                - Activation(activ)
                - ConvTranspose2d(knum[0], input_shape[0], kernel_size=ksize[0], stride=kstride[0], padding=padding, output_padding=1)
                - Dropout(Ctdroprate[0])
                - Activation(activ)

    """
    def __init__(self, ksize=[3,3,3,3,3,3], knum=[256,512,1024],
                 kstride=[2,2,2,2,2,2], Cdroprate=[0.2,0.2,0.2,0.2,0.2,0.2],Ctdroprate=[0.,0.2,0.2,0.2,0.2,0.2],
                  Ddroprate=[0.2,0.2,0.2,0.2,0.2,0.2],Dtdroprate=[0.2,0.2,0.2,0.2,0.2,0.2],z_drop=0.2, activ=nn.Tanh(),
                 padding=1,input_shape=[1,32,32],Dsize=[128,64,32],z_dim=8,bias=False):

        super(AutoEncoder, self).__init__()
        
        self.activ = activ
        self.input_shape=input_shape
        self.z_dim=z_dim
        self.z_drop=z_drop
        self.hidden=knum[-1]

        # Create convolutional and pooling layers
        self.CDP_layers = nn.ModuleList()
        self.CDPt_layers = nn.ModuleList()
        Cnum = len(knum)
        Dnum = len(Dsize)
        ksize=ksize[:Cnum]
        kstride=kstride[:Cnum]
        Cdroprate=Cdroprate[:Cnum]
        Ctdroprate=Ctdroprate[:Cnum]
        Ddroprate=Ddroprate[:Dnum]
        Dtdroprate=Dtdroprate[:Dnum]
        if z_drop!=0.:
            Ctdroprate[0]=0.
        for i in range(Cnum):
            self.CDP_layers.append(nn.Conv2d(self.input_shape[0] if i == 0 else knum[i-1], knum[i], kernel_size=ksize[i], stride=kstride[i], padding=padding,bias=False))
            self.CDP_layers.append(nn.Dropout(Cdroprate[i],inplace=False))
            self.CDP_layers.append(self.activ)

            # if psize[i] is not None:
            #     if ptype == 'max':
            #         self.CDP_layers.append(nn.MaxPool2d(kernel_size=psize[i], stride=pstride[i],padding=ppadding))
            #     else:
            #         self.pool_layers.append(nn.AvgPool2d(kernel_size=psize[i], stride=pstride[i],padding=ppadding))
        for i in range(Cnum):
            self.CDPt_layers.append(nn.ConvTranspose2d(knum[Cnum-i-1], self.input_shape[0] if i == Cnum-1 else knum[Cnum-i-2], kernel_size=ksize[Cnum-i-1], stride=kstride[Cnum-i-1],padding=padding,output_padding=1))
            self.CDPt_layers.append(nn.Dropout(Ctdroprate[Cnum-i-1],inplace=False))
            self.CDPt_layers.append(self.activ)            
            

        self.Dlayers = nn.ModuleList()
        self.Dtlayers = nn.ModuleList()
        self.out_dim,self.out_n_pix = self._get_flattened_size()
        in_features=self.out_dim
        
        for i in range(Dnum):
            D=Dsize[i]
            self.Dlayers.append(nn.Linear(in_features, D))
            self.Dlayers.append(nn.Dropout(p=Ddroprate[i]))            
            self.Dlayers.append(self.activ)            
            in_features = D
        self.outD=nn.Linear(in_features,self.z_dim)
        self.inD=nn.Linear(self.z_dim,in_features)
        for i in range(len(Dsize)):
            D=Dsize[Dnum-i-2] if i!=Dnum-1 else self.out_dim
            self.Dtlayers.append(nn.Linear(in_features,D))
            self.Dtlayers.append(nn.Dropout(p=Dtdroprate[Dnum-i-1]))            
            self.Dtlayers.append(self.activ)                    
            in_features = D
        self.flatten = nn.Flatten()


    def _get_flattened_size(self):
        # Determine the size after flattening
        with torch.no_grad():
            x = torch.zeros(self.input_shape)
            for CDPlayer in self.CDP_layers:
                x=CDPlayer(x)
            return x.numel(),x.shape[1]

    def encoder(self,x):
        for CDPlayer in self.CDP_layers:
            x=CDPlayer(x)
        x=self.flatten(x)
        for Dlayer in self.Dlayers:
            x = Dlayer(x)
        z=self.outD(x)
        return z
    def decoder(self,z):
        _x=(self.inD(z))
        _x=self.activ(nn.Dropout(p=self.z_drop)(_x))
        for Dlayer in self.Dtlayers:
            _x=Dlayer(_x)
        _x = _x.view(-1, self.hidden, self.out_n_pix, self.out_n_pix)
        for CDPtlayer in self.CDPt_layers:
            _x=CDPtlayer(_x)
        return _x
    def forward(self,x):
        z=self.encoder(x)
        _x=self.decoder(z)
        return _x,z
    
    
class net_s(nn.Module):
    def __init__(self, inp_shape, Dsize, out_shape, activ_mid):
        super(net_s, self).__init__()
        layers = []
        layers.append(nn.Linear(inp_shape, Dsize[0]))
        layers.append(activ_mid)
        for i in range(len(Dsize) - 1):
            layers.append(nn.Linear(Dsize[i], Dsize[i+1]))
            layers.append(activ_mid)
        layers.append(nn.Linear(Dsize[-1], out_shape))
        self.Dlayers = nn.Sequential(*layers)

    def forward(self, z):
        return self.Dlayers(z)

class net_t(nn.Module):
    def __init__(self, inp_shape, Dsize, out_shape, activ_mid):
        super(net_t, self).__init__()
        layers = []
        layers.append(nn.Linear(inp_shape, Dsize[0]))
        layers.append(activ_mid)
        for i in range(len(Dsize) - 1):
            layers.append(nn.Linear(Dsize[i], Dsize[i+1]))
            layers.append(activ_mid)
        layers.append(nn.Linear(Dsize[-1], out_shape))
        self.Dlayers = nn.Sequential(*layers)

    def forward(self, z):
        return self.Dlayers(z)



class NVP(nn.Module):
    def __init__(self,inp_shape_s=3,Dsize_s=[32,32,32],out_shape_s=2,activ_mid_s=nn.LeakyReLU(),
                 inp_shape_t=3,Dsize_t=[32,32,32],out_shape_t=2,activ_mid_t=nn.LeakyReLU(),
                 net_s=net_s,net_t=net_t,masks=torch.as_tensor(np.array([[0,1],[1,0]]*3)),prior=distributions.MultivariateNormal(
    torch.zeros(2), torch.eye(2))
):
        super(NVP, self).__init__()
        # Base distribution, a data-dimensional Gaussian
        self.prior = prior
        # Masks are not to be optimised
        self.masks = nn.Parameter(masks, requires_grad=False)
        # The s and t nets that parameterise the scale and shift
        # change of variables according to inputs, here we are
        # duplicating the networks for each layer.
        self.s = torch.nn.ModuleList(
            [net_s(inp_shape=inp_shape_s, Dsize=Dsize_s, out_shape=out_shape_s, activ_mid=activ_mid_s)
             for _ in range(len(masks))]
        )
        self.t = torch.nn.ModuleList(
            [net_t(inp_shape=inp_shape_t, Dsize=Dsize_t, out_shape=out_shape_t, activ_mid=activ_mid_t)
             for _ in range(len(masks))]
        )
    def reverse(self, z, y):
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(len(self.t)):
            x_ = x * self.masks[i]
            s = self.s[i](torch.cat([x_, y], 1)) * (1 - self.masks[i])
            t = self.t[i](torch.cat([x_, y], 1)) * (1 - self.masks[i])
            x = x_ + (1 - self.masks[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def forward(self, x, y):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.masks[i] * z
            s = self.s[i](torch.cat([z_, y], 1)) * (1 - self.masks[i])
            t = self.t[i](torch.cat([z_, y], 1)) * (1 - self.masks[i])
            z = (1 - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self, x, y):
        z, logp = self.reverse(x, y)
        return self.prior.log_prob(z) + logp
        
    def sample(self, n, y): 
        z = self.prior.sample((n,))
        logp = self.prior.log_prob(z)
        x, _ = self.reverse(z, y)
        return x
