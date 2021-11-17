
import math
import numbers
import sys
import os
import torch
import torch.nn as nn
from torch import Tensor
import itertools
import argparse
from enum import Enum
from collections.abc import Sequence
from typing import List, Tuple, Any, Optional

import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.nn import Dropout2d
from torch.utils import data

from det_ae import AutoEncoder as DAutoEncoder
from det_vae import VAE as DVAE
from det_vae import Encoder as DVAEEncoder
from det_vae import Decoder as DVAEDecoder
from task_detector import TaskDetector

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock, ProgConv2DBNBlock, ProgConvTranspose2DBNBlock

NN_SIZE = (64, 64)
Z_DIM = 128
DZ_DIM = 64
DH_DIM = 300


def loadPrognet(prognet, colList, filepath):
    for t in colList:
        prognet.addColumn(msg = t)
    lsd = torch.load(filepath)
    prognet.load_state_dict(lsd)
    return prognet




def invert(img: Tensor) -> Tensor:
    if img.dim() < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.dim()))
    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img

def edgey(img):
    filter = torch.tensor([[-1.0, -1.0 , -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
    f = filter.expand(1,3,3,3)
    res = F.conv2d(img, f, stride = 1, padding = 1)
    res = torch.cat((res, res, res), dim = 1)
    return res



def transformInput(x, method, device, maskIter = None, persTransformer = None):
    with torch.no_grad():
        if method == "denoise":
            xOrig = x.clone()
            xTran = x + torch.randn(*x.shape) * 0.2
        elif method == "colorize":
            xOrig = x.clone()
            img = x.mean(dim = (1,), keepdim = True)
            xTran = torch.cat((img, img, img), dim = 1)
        elif method == "inpaint":
            if maskIter is None:
                raise ValueError("You must supply a maskIter for inpainting.")
            xOrig = x.clone()
            mask, _ = next(maskIter)
            xTran = x * mask
        elif method == "perspective":
            if persTransformer is None:
                raise ValueError("You must supply a persTransformer for perspective.")
            xOrig = x.clone()
            xTran = persTransformer(x)
        elif method == "vflip_to_hflip":
            xOrig = torch.flip(x.clone(), (3,))
            xTran = torch.flip(x, (2,))
        elif method == "invert_flip":
            xOrig = torch.flip(x.clone(), (2,))
            xTran = invert(x)
        elif method == "edge_double_flip":
            xOrig = torch.flip(torch.flip(x.clone(), (3,)), (2,))
            xTran = edgey(x)
        elif method == "doom":
            xOrig = x.clone()
            xTran = invert(edgey(torch.flip(torch.flip(x, (3,)), (2,)) + torch.randn(*x.shape) * 0.2))
    return (xOrig.to(device), xTran.to(device))




class DetGen:
    def __init__(self, aeMode, device):
        super().__init__()
        self.aeMode = aeMode
        self.dev = device

    def generateDetector(self):
        if self.aeMode == "vae":
            enc = DVAEEncoder(NN_SIZE, DZ_DIM, DZ_DIM, h = DH_DIM)
            dec = DVAEDecoder(Z_DIM, NN_SIZE, h = DH_DIM)
            vae = DVAE(enc, dec)
            return vae.to(self.dev)
        else:
            return DAutoEncoder(NN_SIZE, DH_DIM, DZ_DIM).to(self.dev)




class VAEColumn(ProgColumn):
    def __init__(self, colID, blockList, varBlock, parentCols = []):
        super().__init__(colID, blockList, parentCols)
        self.varBlock = (varBlock,)

    def getDist(self):
        return (self.varBlock[0].mu, self.varBlock[0].var)





class ProgVariationalBlock(ProgInertBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module1 = nn.Linear(inSize, outSize)
        self.module2 = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.mu = None
        self.var = None
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        mu = self.module1(x)
        var = self.module2(x)
        return (mu, var)

    def runActivation(self, x):
        mu, var = x
        self.mu = mu
        self.var = var
        return (mu, var)

    def getData(self):
        data = dict()
        data["type"] = "Variational"
        data["input_sizes"] = [self.inSize]
        data["output_sizes"] = [self.outSize, self.outSize]
        return data

    def getShape(self):
        return (self.inSize, self.outSize)




class DropBlock2D(ProgInertBlock):
    def __init__(self, p = 0.2):
        super().__init__()
        self.mod = Dropout2d(p = p)

    def runBlock(self, x):
        return x

    def runActivation(self, x):
        return self.mod(x)

    def getData(self):
        data = dict()
        data["type"] = "Drop2D"
        return data

    def getShape(self):
        return (None, None)




class VAEModelGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def reparamaterize(self, x):
        mu, log_var = x
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def generateColumn(self, parentCols, msg = None):
        cols = []
        cols.append(ProgConv2DBNBlock(3, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(32, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(64, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(128, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(256, 512, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgLambdaBlock(512, 512 * 4, lambda x: torch.flatten(x, start_dim=1)))
        vb = ProgVariationalBlock(512 * 4, Z_DIM, len(parentCols))
        cols.append(vb)
        cols.append(ProgLambdaBlock(Z_DIM, Z_DIM, self.reparamaterize))
        cols.append(ProgDenseBlock(Z_DIM, 512 * 4, len(parentCols), activation=None))
        cols.append(ProgLambdaBlock(512 * 4, 512, lambda x: x.view(-1, 512, 2, 2)))
        cols.append(ProgConvTranspose2DBNBlock(512, 256, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(256, 128, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(128, 64, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(64, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(32, 32, 3, len(parentCols), activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(32, 3, 3, len(parentCols), activation=nn.Tanh(), layerArgs={'padding': 1}))
        return VAEColumn(self._genID(), cols, vb, parentCols = parentCols)

    def _genID(self):
        id = self.ids
        self.ids += 1
        return id




class VAEExpertGenerator(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def reparamaterize(self, x):
        mu, log_var = x
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def generateColumn(self, parentCols, msg = None):
        cols = []
        cols.append(ProgConv2DBNBlock(3, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(32, 64, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(64, 128, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(128, 256, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(256, 512, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1}))
        cols.append(ProgLambdaBlock(512, 512 * 4, lambda x: torch.flatten(x, start_dim=1)))
        cols.append(ProgVariationalBlock(512 * 4, Z_DIM, 0))
        cols.append(ProgLambdaBlock(Z_DIM, Z_DIM, self.reparamaterize))
        cols.append(ProgDenseBlock(Z_DIM, 512 * 4, 0, activation=None))
        cols.append(ProgLambdaBlock(512 * 4, 512, lambda x: x.view(-1, 512, 2, 2)))
        cols.append(ProgConvTranspose2DBNBlock(512, 256, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(256, 128, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(128, 64, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(64, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConvTranspose2DBNBlock(32, 32, 3, 0, activation=nn.LeakyReLU(), layerArgs={'stride': 2, 'padding': 1, 'output_padding': 1}))
        cols.append(DropBlock2D())
        cols.append(ProgConv2DBNBlock(32, 3, 3, 0, activation=nn.Tanh(), layerArgs={'padding': 1}))
        return ProgColumn(self._genID(), cols, parentCols = [])

    def _genID(self):
        id = self.ids
        self.ids += 1
        return id







#===============================================================================
