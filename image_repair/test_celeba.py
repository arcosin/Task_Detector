
import sys
import os
import json
import torch
import torch.nn as nn
import itertools
import argparse
from collections import defaultdict

import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils import data

from Doric import ProgNet, ProgColumn, ProgColumnGenerator
from Doric import ProgDenseBlock, ProgLambdaBlock, ProgInertBlock, ProgDeformConv2DBlock, ProgDeformConv2DBNBlock, ProgConvTranspose2DBNBlock
from task_detector import TaskDetector
import common

TASKS = ["denoise", "colorize", "inpaint", "perspective", "vflip_to_hflip", "invert_flip", "edge_double_flip"]






def printCM(cm):
    print("          Predicted:")
    print("         ", end = '')
    dotsLen = 0
    for task in TASKS:
        s = task + "   "
        print(s, end = '')
        dotsLen += len(s)
    print()
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()
    for taskNum in range(len(TASKS)):
        print("Task " + str(taskNum) + ":  |", end = '')
        for predNum in range(len(TASKS)):
            print("{:8.3f}".format(cm[(TASKS[predNum], TASKS[taskNum])]), end = "   ")
        print("|")
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()





def test(args, model, data, task, device):
    model.eval()
    logDict = {"mse": []}
    with torch.no_grad():
        totalLoss = 0.0
        mli = None
        rp = None
        if task == "inpaint":
            mds = datasets.ImageFolder(args.mask_dir, transforms.ToTensor())
            mld = torch.utils.data.DataLoader(mds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            mli = itertools.cycle(mld)
        if task == "perspective":
            rp = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        print("Testing %s." % args.algo)
        for i, (x, _) in enumerate(data):
            try:
                xOrig, x = common.transformInput(x, task, device, maskIter = mli, persTransformer = rp)
                xReconst = model(x)
                reconstLoss = F.mse_loss(xReconst, xOrig).item()
                print("   Test %s img %d:   %s." % (task, i, reconstLoss))
                logDict["mse"].append(reconstLoss)
                totalLoss += reconstLoss
                xCat = torch.cat([xOrig.cpu().data, x.cpu().data, xReconst.cpu().data], dim=3)
                save_image(xCat, os.path.join(args.output_dir, 'test-{}-{}-{}.png'.format(args.algo, task, i)))
            except:
                print("   Test %s img %d:   %s." % (task, i, "ERROR."))
        print("Total test loss:  %f." % totalLoss)
        print("Avg test loss:  %f." % (totalLoss / len(data)))
        with open(os.path.join(args.load_dir, "%s_test_log_%s.json" % (args.algo, task)), 'w') as jsonfile:
            json.dump(logDict, jsonfile)



def testProg(args, model, data, task, device):
    model.eval()
    logDict = {"mse": []}
    with torch.no_grad():
        totalLoss = 0.0
        mli = None
        rp = None
        if task == "inpaint":
            mds = datasets.ImageFolder(args.mask_dir, transforms.ToTensor())
            mld = torch.utils.data.DataLoader(mds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            mli = itertools.cycle(mld)
        if task == "perspective":
            rp = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        print("Testing %s." % args.algo)
        for i, (x, _) in enumerate(data):
            try:
                xOrig, x = common.transformInput(x, task, device, maskIter = mli, persTransformer = rp)
                ind = TASKS.index(task)
                xReconst = model(ind, x)
                reconstLoss = F.mse_loss(xReconst, xOrig).item()
                print("   C Test %s img %d:   %s." % (task, i, reconstLoss))
                logDict["mse"].append(reconstLoss)
                totalLoss += reconstLoss
                xCat = torch.cat([xOrig.cpu().data, x.cpu().data, xReconst.cpu().data], dim=3)
                save_image(xCat, os.path.join(args.output_dir, 'test-{}-{}-{}.png'.format(args.algo, task, i)))
            except:
                print("   Test %s img %d:   %s." % (task, i, "ERROR."))
        print("Total test loss:  %f." % totalLoss)
        print("Avg test loss:  %f." % (totalLoss / len(data)))
        with open(os.path.join(args.load_dir, "%s_test_log_%s.json" % (args.algo, task)), 'w') as jsonfile:
            json.dump(logDict, jsonfile)



def testTD(args, model, td, data, task, device, cm = defaultdict(lambda: 0.0)):
    model.eval()
    logDict = {"mse": []}
    with torch.no_grad():
        totalLoss = 0.0
        mli = None
        rp = None
        if task == "inpaint":
            mds = datasets.ImageFolder(args.mask_dir, transforms.ToTensor())
            mld = torch.utils.data.DataLoader(mds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            mli = itertools.cycle(mld)
        if task == "perspective":
            rp = transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
        print("Testing %s." % args.algo)
        for i, (x, _) in enumerate(data):
            try:
                xOrig, x = common.transformInput(x, task, device, maskIter = mli, persTransformer = rp)
                dtask, _ = td.detect(x)
                fscores, _ = td.getLastScores()
                cm[(dtask, task)] += 1
                ind = TASKS.index(task)
                xReconst = model(ind, x)
                reconstLoss = F.mse_loss(xReconst, xOrig).item()
                print("   Test %s img %d:   %s." % (task, i, reconstLoss))
                print("      Familiarity scores:  %s." % str(fscores))
                logDict["mse"].append(reconstLoss)
                totalLoss += reconstLoss
                xCat = torch.cat([xOrig.cpu().data, x.cpu().data, xReconst.cpu().data], dim=3)
                save_image(xCat, os.path.join(args.output_dir, 'test-{}-{}-{}.png'.format(args.algo, task, i)))
            except:
                print("   Test %s img %d:   %s." % (task, i, "ERROR."))
        print("Total test loss:  %f." % totalLoss)
        print("Avg test loss:  %f." % (totalLoss / len(data)))
        with open(os.path.join(args.load_dir, "%s_test_log_%s.json" % (args.algo, task)), 'w') as jsonfile:
            json.dump(logDict, jsonfile)
    return cm





def main(args):
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda:0')
    print("Using device:  %s." % device)
    transform = transforms.Compose([transforms.CenterCrop(148), transforms.Resize(64), transforms.ToTensor()])
    dataset = datasets.ImageFolder(args.test_dir, transform)
    dl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    if args.algo == "base" or args.algo == "ewc":
        loadPath = os.path.join(args.load_dir, "%s_model_%s.pt" % (args.algo, TASKS[-1]))
        gen = common.VAEExpertGenerator()
        model = gen.generateColumn([])
        model.load_state_dict(torch.load(loadPath))
        model.to(device)
        for t in TASKS:
            test(args, model, dl, t, device)
    elif args.algo == "mom":
        loadPath = os.path.join(args.load_dir, "mixmod_model_%s.pt" % TASKS[-1])
        model = ProgNet(colGen=common.VAEExpertGenerator())
        model = common.loadPrognet(model, TASKS, loadPath)
        model.to(device)
        for t in TASKS:
            testProg(args, model, dl, t, device)
    elif args.algo == "mom_td":
        loadPath = os.path.join(args.load_dir, "mixmod_model_%s.pt" % TASKS[-1])
        model = ProgNet(colGen=common.VAEExpertGenerator())
        model = common.loadPrognet(model, TASKS, loadPath)
        model.to(device)
        td = TaskDetector(common.DetGen(args.dt_type, device), args.load_dir)
        td.loadAll(TASKS)
        td.to(device)
        cm = defaultdict(lambda: 0.0)
        for t in TASKS:
            cm = testTD(args, model, td, dl, t, device, cm)
        printCM(cm)
    elif args.algo == "prognet":
        loadPath = os.path.join(args.load_dir, "prognet_model_%s.pt" % TASKS[-1])
        model = ProgNet(colGen=common.VAEModelGenerator())
        model = common.loadPrognet(model, TASKS, loadPath)
        model.to(device)
        for t in TASKS:
            testProg(args, model, dl, t, device)
    elif args.algo == "prognet_td":
        loadPath = os.path.join(args.load_dir, "prognet_model_%s.pt" % TASKS[-1])
        model = ProgNet(colGen=common.VAEModelGenerator())
        model = common.loadPrognet(model, TASKS, loadPath)
        model.to(device)
        td = TaskDetector(common.DetGen(args.dt_type, device), args.load_dir)
        td.loadAll(TASKS)
        td.to(device)
        cm = defaultdict(lambda: 0.0)
        for t in TASKS:
            cm = testTD(args, model, td, dl, t, device, cm)
        printCM(cm)





def configCLIParser(parser):
    parser.add_argument("--cpu", help="Specify whether the CPU should be used.", type=bool, nargs='?', const=True, default=False)
    parser.add_argument("--algo", help="Algorithm to test.", choices=["base", "ewc", "mom", "mom_td", "prognet", "prognet_td"], default="prognet_td")
    parser.add_argument("--dt_type", help="Type of AE detector to use.", choices=["ae", "vae"], default="ae")
    parser.add_argument("--output_dir", help="Specify where to log the output to.", type=str, default="./data/outputs/")
    parser.add_argument("--load_dir", help="Specify path to save model.", type=str, default="./data/models/")
    parser.add_argument("--test_dir", help="Specify testing set directory.", type=str, default="./data/celeba_small_test/")
    parser.add_argument("--mask_dir", help="Specify mask set directory.", type=str, default="./data/mask/")
    parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
    return parser



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = "", description = "")
    parser = configCLIParser(parser)
    args = parser.parse_args()
    main(args)

#===============================================================================
