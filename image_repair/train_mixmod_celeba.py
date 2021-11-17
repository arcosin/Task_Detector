

import os
import json
import torch
import torch.nn as nn
from torch.nn import Dropout2d
import itertools
import argparse

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils import data

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock, ProgConv2DBNBlock, ProgConvTranspose2DBNBlock
import common


z_dim = 128

g_mu = None
g_var = None

mask_dataset = None
mask_loader = None
mask_loader_iter = None
perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)



def transform_input(x, method, device):
    with torch.no_grad():
        if method == "denoise":
            xOrig = x.clone()
            xTran = x + torch.randn(*x.shape) * 0.2
        elif method == "colorize":
            xOrig = x.clone()
            img = x.mean(dim = (1,), keepdim = True)
            xTran = torch.cat((img, img, img), dim = 1)
        elif method == "inpaint":
            xOrig = x.clone()
            mask, _ = next(mask_loader_iter)
            xTran = x * mask
        elif method == "perspective":
            xOrig = x.clone()
            xTran = perspective_transformer(x)
        elif method == "vflip_to_hflip":
            xOrig = torch.flip(x.clone(), (3,))
            xTran = torch.flip(x, (2,))
        elif method == "invert_flip":
            xOrig = torch.flip(x.clone(), (2,))
            xTran = common.invert(x)
        elif method == "edge_double_flip":
            xOrig = torch.flip(torch.flip(x.clone(), (3,)), (2,))
            xTran = common.edgey(x)
        elif method == "doom":
            xOrig = x.clone()
            xTran = common.invert(torch.flip(torch.flip(x, (3,)), (2,)) + torch.randn(*x.shape) * 0.2)
    return (xOrig.to(device), xTran.to(device))







class ProgVariationalBlock(ProgInertBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module1 = nn.Linear(inSize, outSize)
        self.module2 = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        mu = self.module1(x)
        var = self.module2(x)
        return (mu, var)

    def runActivation(self, x):
        global g_mu
        global g_var
        mu, var = x
        g_mu = mu
        g_var = var
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





class VariationalAutoEncoderModelGenerator(ProgColumnGenerator):
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
        cols.append(ProgVariationalBlock(512 * 4, z_dim, 0))
        cols.append(ProgLambdaBlock(z_dim, z_dim, self.reparamaterize))
        cols.append(ProgDenseBlock(z_dim, 512 * 4, 0, activation=None))
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
        return ProgColumn(self.__genID(), cols, parentCols = [])

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id






def train(model, batch_size, epochs, device, transform_method, trainDir, modelDir, outDir, bs, lr, kl_weight=3e-5, copyTask=None):
    global g_mu, g_var
    lossDict = {"rec": [], "kl": [], "all": []}
    col = model.addColumn()
    if copyTask is not None:
        currCol = model.getColumn(col)
        copyCol = model.getColumn(copyTask)
        currCol.load_state_dict(copyCol.state_dict())
    model = model.to(device)
    model.train()
    transform = transforms.Compose([transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor()])
    dataset = datasets.ImageFolder(trainDir, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, (x, _) in enumerate(data_loader):
            xOrig, x = transform_input(x, transform_method, device)
            x_reconst = model(col, x)
            reconst_loss = F.mse_loss(x_reconst, xOrig)
            lossDict["rec"].append(reconst_loss.item())
            kl_div = torch.mean(-0.5 * torch.sum(1 + g_var - g_mu ** 2 - g_var.exp(), dim = 1), dim = 0)
            lossDict["kl"].append(kl_div.item())
            loss = reconst_loss + kl_weight * kl_div
            lossDict["all"].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ("Col: {}, Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}, Loss: {:.4f}".format(col, epoch+1, epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item(), loss.item()))
        with torch.no_grad():
            x_concat = torch.cat([xOrig.cpu().data, x.cpu().data, x_reconst.cpu().data], dim=3)
            save_image(x_concat, os.path.join(outDir, 'mixmod-col-%s-reconst-%d.png' % (col, epoch+1)))
    torch.save(model.state_dict(), os.path.join(modelDir, "mixmod_model_%s.pt" % transform_method))
    with open(os.path.join(modelDir, "mixmod_losses_%s.json" % transform_method), 'w') as jsonfile:
        json.dump(lossDict, jsonfile)
    dataset = None
    data_loader = None
    return col




def main(args):
    global mask_dataset, mask_loader, mask_loader_iter
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print("Using device:  %s." % device)
    model = ProgNet(colGen=VariationalAutoEncoderModelGenerator())
    print("Training started.")
    train(model, args.batch_size, args.epochs, device, "denoise", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr)
    model.freezeAllColumns()
    train(model, args.batch_size, args.epochs, device, "colorize", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, copyTask=0)
    model.freezeAllColumns()
    mask_dataset = datasets.ImageFolder(args.mask_dir, transforms.ToTensor())
    mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    mask_loader_iter = itertools.cycle(mask_loader)
    train(model, args.batch_size, args.epochs, device, "inpaint", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, kl_weight=3e-6, copyTask=1)
    model.freezeAllColumns()
    train(model, args.batch_size, args.epochs, device, "perspective", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, copyTask=2)
    model.freezeAllColumns()
    train(model, args.batch_size, args.epochs, device, "vflip_to_hflip", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, copyTask=3)
    model.freezeAllColumns()
    train(model, args.batch_size, args.epochs, device, "invert_flip", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, copyTask=4)
    model.freezeAllColumns()
    train(model, args.batch_size, args.epochs, device, "edge_double_flip", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, copyTask=5)
    model.freezeAllColumns()
    #train(model, args.batch_size, args.epochs, device, "doom", args.train_dir, args.save_dir, args.output_dir, args.batch_size, args.lr, copyTask=6)
    print("Training done.")




def configCLIParser(parser):
    parser.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--output_dir", help="Specify where to log the output to.", type=str, default="./data/outputs/")
    parser.add_argument("--save_dir", help="Specify path to save model.", type=str, default="./data/models/")
    parser.add_argument("--train_dir", help="Specify training set directory.", type=str, default="./data/celeba_small_train/")
    parser.add_argument("--mask_dir", help="Specify mask set directory.", type=str, default="./data/mask/")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=32)
    parser.add_argument("--epochs", help="Epochs.", type=int, default=10)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=0.0005)
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "", description = "")
    parser = configCLIParser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
