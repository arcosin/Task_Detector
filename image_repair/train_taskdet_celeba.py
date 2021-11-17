
import os
import json
import torch
import torch.nn as nn
import itertools
import argparse

import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import save_image

from det_ae import AutoEncoder
from det_vae import VAE
from det_vae import Encoder as VAEEncoder
from det_vae import Decoder as VAEDecoder
from task_detector import TaskDetector

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock, ProgDeformConv2DBlock, ProgDeformConv2DBNBlock, ProgConvTranspose2DBNBlock
import common

Z_DIM = 64
NN_SIZE = (64, 64)
H_DIM = 300

g_mu = None
g_var = None

mask_dataset = None
mask_loader = None
mask_loader_iter = None
perspective_transformer = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)



class DetGen:
    def __init__(self, aeMode, device):
        super().__init__()
        self.aeMode = aeMode
        self.dev = device

    def generateDetector(self):
        if self.aeMode == "vae":
            enc = VAEEncoder(NN_SIZE, Z_DIM, Z_DIM, h = H_DIM)
            dec = VAEDecoder(Z_DIM, NN_SIZE, h = H_DIM)
            vae = VAE(enc, dec)
            return vae.to(self.dev)
        else:
            return AutoEncoder(NN_SIZE, H_DIM, Z_DIM).to(self.dev)



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







def train(model, epochs, device, transform_method, trainDir, modelDir, bs):
    lossDict = {"rec": [], "kl": [], "all": []}
    model.addTask(transform_method)
    transform = transforms.Compose([transforms.CenterCrop(148),
                                            transforms.Resize(64),
                                            transforms.ToTensor()])
    dataset = datasets.ImageFolder(trainDir, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    for epoch in range(epochs):
        for i, (x, _) in enumerate(data_loader):
            _, x = transform_input(x, transform_method, device)
            _, loss = model.trainStep(x, transform_method)
            if isinstance(loss, dict):
                lossDict["rec"].append(loss["reconst"])
                lossDict["kl"].append(loss["diverge"])
                lossDict["all"].append(loss["loss"])
                print("Env: {}  Epoch: {}  Image: {}/{}  Loss: {}.".format(transform_method, epoch, i, len(data_loader), loss["loss"]))
            else:
                lossDict["all"].append(loss)
                print("Env: {}  Epoch: {}  Image: {}/{}  Loss: {}.".format(transform_method, epoch, i, len(data_loader), loss))
    with open(os.path.join(modelDir, "taskdet_losses_%s.json" % transform_method), 'w') as jsonfile:
        json.dump(lossDict, jsonfile)
    model.expelDetector(transform_method)
    dataset = None
    data_loader = None




def main(args):
    global mask_dataset, mask_loader, mask_loader_iter
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print("Using device:  %s." % device)
    gen = DetGen(args.ae_type, device)
    model = TaskDetector(gen, args.save_dir)
    print("Training started.")
    train(model, args.epochs, device, "denoise", args.train_dir, args.save_dir, args.batch_size)
    train(model, args.epochs, device, "colorize", args.train_dir, args.save_dir, args.batch_size)
    mask_dataset = datasets.ImageFolder(args.mask_dir, transforms.ToTensor())
    mask_loader = torch.utils.data.DataLoader(mask_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    mask_loader_iter = itertools.cycle(mask_loader)
    train(model, args.epochs, device, "inpaint", args.train_dir, args.save_dir, args.batch_size)
    train(model, args.epochs, device, "perspective", args.train_dir, args.save_dir, args.batch_size)
    train(model, args.epochs, device, "vflip_to_hflip", args.train_dir, args.save_dir, args.batch_size)
    train(model, args.epochs, device, "invert_flip", args.train_dir, args.save_dir, args.batch_size)
    train(model, args.epochs, device, "edge_double_flip", args.train_dir, args.save_dir, args.batch_size)
    #train(model, args.epochs, device, "doom", args.train_dir, args.save_dir, args.batch_size)
    print("Training done.")




def configCLIParser(parser):
    parser.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--ae_type", help="Type of AE to use.", choices=["ae", "vae"], default="ae")
    parser.add_argument("--save_dir", help="Specify path to save model.", type=str, default="./data/models/")
    parser.add_argument("--train_dir", help="Specify training set directory.", type=str, default="./data/celeba_small_train/")
    parser.add_argument("--mask_dir", help="Specify mask set directory.", type=str, default="./data/mask/")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=32)
    parser.add_argument("--epochs", help="Epochs.", type=int, default=15)
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "", description = "")
    parser = configCLIParser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
