
import sys
import time
import random
import json
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image as Image
from datetime import datetime

from .multienv import MultiEnv
from .random_agent import RandomAgent
from .task_detector import TaskDetector

from .autoencoder import AutoEncoder

from .vae import VAE
from .vae import Encoder as VAEEncoder
from .vae import Decoder as VAEDecoder

from .aae import AAE
from .aae import Encoder as AAEEncoder
from .aae import Decoder as AAEDecoder
from .aae import Descriminator as AAEDescriminator

from .baseline_classifier import TaskClassifier


GPU_TRAINING_ON = True
TRAIN_RECS = 1000
TRAIN_EPOCHS = 1
TEST_RECS = 100

NN_SIZE = (77, 100)
H_DIM = 300
AAED_H_DIM_1 = 128
AAED_H_DIM_2 = 32
Z_DIM = 128

DEF_ENVS = ["breakout", "pong", "space_invaders", "ms_pacman", "assault", "asteroids", "boxing", "phoenix", "alien"]

device = None

SAMPLES_DIR = 'samples'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'




class DetGen:
    def __init__(self, aeMode):
        super().__init__()
        self.aeMode = aeMode

    def generateDetector(self):
        if self.aeMode == "vae":
            vae = buildVAE()
            return vae.to(device)
        elif self.aeMode == "aae":
            aae = buildAAE()
            return aae.to(device)
        else:
            return AutoEncoder(NN_SIZE, H_DIM, Z_DIM).to(device)


# Source:  https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745.
class AddGaussianNoise:
    def __init__(self, mean = 0.0, std = 0.5):
        self.std = std
        self.mean = mean

    def __call__(self, t):
        return t + (torch.randn(t.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def buildVAE():
    enc = VAEEncoder(NN_SIZE, Z_DIM, Z_DIM, h = H_DIM)
    dec = VAEDecoder(Z_DIM, NN_SIZE, h = H_DIM)
    model = VAE(enc, dec)
    return model


def buildAAE():
    enc = AAEEncoder(NN_SIZE, Z_DIM, h = H_DIM)
    dec = AAEDecoder(Z_DIM, NN_SIZE, h = H_DIM)
    #des = AAEDescriminator(Z_DIM, h1 = AAED_H_DIM_1, h2 = AAED_H_DIM_2)
    des = AAEDescriminator(h1 = AAED_H_DIM_1, h2 = AAED_H_DIM_2)
    model = AAE(enc, dec, des, Z_DIM)
    return model



def preprocess(inputDict, addNoise = False):
    if addNoise:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(NN_SIZE, Image.NEAREST),
            lambda x: transforms.functional.vflip(x),
            transforms.ToTensor(),
            AddGaussianNoise(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(NN_SIZE, Image.NEAREST),
            lambda x: transforms.functional.vflip(x),
            transforms.ToTensor(),
        ])
    x = inputDict['S0'].T
    x = transform(x)
    x = torch.unsqueeze(x, dim=0)
    x = x.to(device)
    n_x = inputDict['S1'].T
    n_x = transform(n_x)
    n_x = torch.unsqueeze(n_x, dim=0)
    n_x = n_x.to(device).detach()
    return x, n_x



def convertTorch(state):
    return torch.from_numpy(state)



def test(agent, detector, env, envID, ds, log, addNoise):
    detector.toggleSaving(False)
    predicteds = defaultdict(lambda: 0.0)
    predictedsNorm = defaultdict(lambda: 0.0)
    with torch.no_grad():
        for inpNum, inp in enumerate(ds):
            ts = time.time()
            x, n_x = preprocess(inp, addNoise = addNoise)
            if inpNum == 0:
                save_image(torch.rot90(x, 3, [2, 3]), "%s-noise.png" % envID)
                break
            envPred, envPredNorm = detector.detect(x)
            predicteds[str((envPred, envID))] += 1
            predictedsNorm[str((envPredNorm, envID))] += 1
            te = time.time()
            print("Env: {}   Record: {}/{}   Corr: {}={}-->{}   NCorr: {}={}-->{}   Time: {}".format(envID, inpNum, len(ds), envPred, envID, (envPred == envID), envPredNorm, envID, (envPredNorm == envID), te - ts))
    detector.toggleSaving(True)
    return predicteds, predictedsNorm





def genDataFromEnv(agent, env, datasetSize, render = False):
    ds = []
    while True:
        state = convertTorch(env.reset())
        terminal = False
        i = 0
        while not terminal:
            if render:   env.render()
            action = agent.act(state)
            nextState, reward, terminal, info = env.step(action)
            nextState = convertTorch(nextState)
            detectorInput = {"S0": state, "S1": nextState, "A": action}
            ds.append(detectorInput)
            state = nextState
            i = i + 1
        if len(ds) >= datasetSize:
            ds = random.sample(ds, datasetSize)
            return ds





def train(agent, detector, env, envID, ds, epochs, sampleDir, log):
    if log is not None:
        lossKey = str(envID) + "_train_loss"
        samplePathKey = str(envID) + "_train_samples"
        log[lossKey] = []
        log[samplePathKey] = []
    for epoch in range(epochs):
        for inpNum, inp in enumerate(ds):
            ts = time.time()
            out, z, loss = trainDetector(detector, inp, envID)
            te = time.time()
            print("Env: {}   Epoch: {}/{}   Record: {}/{}   Loss: {}   Time: {}".format(envID, epoch, epochs, inpNum, len(ds), loss, te - ts), flush=True)
            if log is not None:
                log[lossKey].append(loss)
        if sampleDir[-1] == '/':
            imgPath = "{}env_{}_e_{}.png".format(sampleDir, envID, epoch)
        else:
            imgPath = "{}/env_{}_e_{}.png".format(sampleDir, envID, epoch)
        if log is not None:
            log[samplePathKey].append(imgPath)
        if out is not None:
            save_image(torch.rot90(out, 3, [2, 3]), imgPath)
        #print("Ooo   ", torch.min(z), torch.max(z), torch.mean(z), torch.std(z))
    for inpNum, inp in enumerate(ds):
        trainDetectorDistro(detector, inp, envID)



def trainDetectorDistro(detector, inputDict, envLabel):
    with torch.no_grad():
        x, n_x = preprocess(inputDict)
        detector.trainDistro(x, envLabel)



def trainDetector(detector, inputDict, envLabel):
    x, n_x = preprocess(inputDict)
    return detector.trainStep(x, n_x, envLabel)


def populateCM(taskList, cm, predicteds):
    for trueEnv in taskList:
        for predEnv in taskList:
            cm[str((predEnv, trueEnv))] += predicteds[str((predEnv, trueEnv))]



def printCM(taskList, cm):
    print("          Predicted:")
    print("         ", end = '')
    dotsLen = 0
    for task in taskList:
        s = task + "   "
        print(s, end = '')
        dotsLen += len(s)
    print()
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()
    for i, trueEnv in enumerate(taskList):
        print("Task " + str(i) + ":  |", end = '')
        for predEnv in taskList:
            print("{:8.3f}".format(cm[str((predEnv, trueEnv))]), end = ' ')
        print("|")
    print("         ", end = '')
    for _ in range(dotsLen):
        print('-', end = '')
    print()



def writeLog(log, filepath):
    with open(filepath, "w") as f:
        json.dump(log, f)



def configCLIParser(parser):
    parser.add_argument("--train_size", help="Number of records to generate for training.", type=int, default=TRAIN_RECS)
    parser.add_argument("--train_epochs", help="Training epochs.", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--test_size", help="Number of records to generate for testing.", type=int, default=TEST_RECS)
    parser.add_argument("--train_mode", help="If 2, train on one enviroment specified by train_env. If 1, trains all detectors. If 0, attempts to load detectors.", choices=[-1, 0, 1, 2], type=int, default=1)
    parser.add_argument("--train_env", help="Env to use if train_mode is set to 2.", choices=DEF_ENVS, default=DEF_ENVS[0])
    parser.add_argument("--test_mode", help="If 1, tests the detectors. If 0, skips testing.", type=int, choices=[0, 1], default=1)
    parser.add_argument("--gen_mode", help="If 1, generates from AE, VAE, or AAE. If 0, skips generation.", type=int, choices=[0, 1], default=0)
    parser.add_argument("--logging", help="Logs important info as JSON.", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ae_type", help="Type of AE to use.", choices=["aae", "vae", "ae", "base"], default="aae")
    parser.add_argument("--detector_cache", help="Size of the detector cache. The default -1 maps to no limit.", type=int, default=-1)
    parser.add_argument("--test_noise", help="If 1, add noise to images during testing.", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", help="Device to run torch on. Usually 'cpu' or 'cuda:[N]'. Defaults to cpu if cuda is not available.", type=str, default="cpu")
    parser.add_argument("--models_dir", help="Directory to store model save / load files.", type=str, default = "./%s/" % MODELS_DIR)
    parser.add_argument("--logs_dir", help="Directory to store JSON log files.", type=str, default = "./%s/" % LOGS_DIR)
    parser.add_argument("--samples_dir", help="Directory to store training reconst samples.", type=str, default = "./%s/" % SAMPLES_DIR)
    return parser



def main(args):
    global device
    if args.logging:
        log = dict()
    else:
        log = None
    print("Starting.", flush=True)
    if torch.cuda.is_available():
        print("Cuda is available.")
        print("Using device: %s." % args.device)
        device = torch.device(args.device)
    else:
        print("Cuda is not available.")
        print("Using device: cpu.")
        device = torch.device("cpu")
    if args.train_mode == 2 or args.train_mode == -1:
        envNameList = [args.train_env]
    else:
        envNameList = DEF_ENVS
    atariGames = MultiEnv(envNameList)
    agent = RandomAgent(atariGames.actSpace)
    if args.ae_type != "base":
        gen = DetGen(args.ae_type)
        taskDetector = TaskDetector(gen, args.models_dir, args.ae_type, detectorCache = args.detector_cache, device = args.device)
    else:
        taskDetector = TaskClassifier(NN_SIZE, len(DEF_ENVS), args.models_dir, h = H_DIM, device = args.device)
    if args.train_mode > 0:
        for i, env in enumerate(atariGames.getEnvList()):
            ds = genDataFromEnv(agent, env, args.train_size)
            print("Mem size of ds now: %s." % sys.getsizeof(ds))
            taskDetector.addTask(env.game)
            print("Mem size of task detector now: %s." % sys.getsizeof(taskDetector))
            train(agent, taskDetector, env, env.game, ds, args.train_epochs, args.samples_dir, log)
            taskDetector.expelDetector(env.game)
        print("Training complete.\n\n")
    else:
        taskDetector.loadAll(envNameList)
        print("Loaded envs %s." % str(envNameList))
    if args.gen_mode == 1 and args.ae_type in ["ae", "vae", "aae"]:
        for n in taskDetector.getNames():
            m = taskDetector.getDetectorModel(n)
            g = m.generate()
            if args.samples_dir[-1] == '/':
                imgPath = "{}env_{}_gen.png".format(args.samples_dir, n)
            else:
                imgPath = "{}env_{}_gen.png".format(args.samples_dir, n)
            save_image(torch.rot90(g, 3, [2, 3]), imgPath)
    if args.test_mode == 1:
        print("Testing with and without normalization.")
        cm = defaultdict(lambda: 0.0)
        cmn = defaultdict(lambda: 0.0)
        for i, env in enumerate(atariGames.getEnvList()):
            ds = genDataFromEnv(agent, env, args.test_size)
            predicteds, predictedsNorm = test(agent, taskDetector, env, env.game, ds, log, addNoise = (args.test_noise == 1))
            populateCM(envNameList, cm, predicteds)
            populateCM(envNameList, cmn, predictedsNorm)
        if log is not None:
            log["env_names"] = envNameList
            log["cm"] = dict(cm)
            log["cmn"] = dict(cmn)
        print("Testing complete.\n")
        print("Not normalized:\n\n")
        printCM(envNameList, cm)
        print("\n\nNormalized:\n\n")
        printCM(envNameList, cmn)
    if log is not None:
        ts = datetime.now().strftime(r"%m-%d-%Y_%H-%M-%S")
        writeLog(log, "{}log-{}-{}".format(args.logs_dir, args.ae_type, ts))
    print("\n\nDone.", flush=True)




if __name__ == '__main__':
    main()

#===============================================================================
