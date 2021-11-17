
import torch
import torch.nn as nn
import torch.nn.functional as F
from det_ae import DAE
from Doric import ProgColumnGenerator
from Doric import ProgConv2DBlock, ProgLambdaBlock, ProgDenseBlock, ProgColumn




def loadPrognet(prognet, colList, filepath):
    for t in colList:
        prognet.addColumn(msg = t)
    prognet.load_state_dict(torch.load(filepath))
    return prognet



class QNetwork(nn.Module):
    def __init__(self, inChannels, actSize):
        super().__init__()
        self.co = nn.Conv2d(inChannels, 16, kernel_size = 3, stride = 1)
        self.fc = nn.Linear(1024, 128)
        self.out = nn.Linear(128, actSize)

    def forward(self, obs):
        x = F.relu(self.co(obs))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        y = self.out(x)
        return y




class ProgWrapper(nn.Module):
    def __init__(self, prognet):
        super().__init__()
        self.prognet = prognet
        self.ids = list(prognet.colMap.keys())
        self.id = None

    def forward(self, obs):
        return self.prognet(self.id, obs)

    def switch(self, id):
        if id in self.ids:
            self.id = id
            self.prognet.freezeAllColumns()
        else:
            self.id = id
            self.prognet.freezeAllColumns()
            self.prognet.addColumn(msg = id)
            self.ids.append(id)




class ProgTD(nn.Module):
    def __init__(self, prognet, td):
        super().__init__()
        self.prognet = prognet
        self.td = td
        self.lastC = None

    def forward(self, obs, c = None):
        if c is None:
            c = self.detect(obs)
        self.lastC = c
        return self.prognet(c, obs)

    def detect(self, obs):
        c, _ = self.td.detect(obs)
        return c

    def getLastCol(self):
        return self.lastC




class DQN:
    def __init__(self, model):
        self.model = model

    def act(self, obs):
        a = self.model(obs).max(1)[1].view(1, 1)
        return a





class QGenerator(ProgColumnGenerator):
    def __init__(self, inChannels, actSize):
        self.ids = 0
        self.inp = inChannels
        self.out = actSize

    def generateColumn(self, parentCols, msg = None):
        cols = []
        cols.append(ProgConv2DBlock(self.inp, 16, 3, len(parentCols), layerArgs = {"stride": 1}))
        cols.append(ProgLambdaBlock(None, None, lambda x: x.view(x.size(0), -1)))
        cols.append(ProgDenseBlock(1024, 128, len(parentCols)))
        cols.append(ProgDenseBlock(128, self.out, len(parentCols), activation = (lambda x: x)))
        return ProgColumn(msg, cols, parentCols = parentCols)




class DetGen:
    def __init__(self, device):
        super().__init__()
        self.dev = device

    def generateDetector(self):
        ae = DAE()
        return ae.to(self.dev)


#===============================================================================
