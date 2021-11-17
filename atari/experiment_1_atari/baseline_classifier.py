
import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class TaskClassifier(nn.Module):
    def __init__(self, inShape, outShape, savePath, h = 300, device = "cpu"):
        super().__init__()
        self.saving = True
        self.dev = device
        self.maxTasks = outShape
        self.taskMap = []
        self.savePath = savePath
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(32)
        ).to(device)
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
        ).to(device)
        self.fch = nn.Linear(inShape[0] * inShape[1] * 128, h).to(device)
        self.fco = nn.Linear(h, outShape).to(device)
        self.lossFunc = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr = 1e-5, weight_decay = 1e-5)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fch(x))
        z = F.softmax(self.fco(x), dim = 1)
        return z

    def detect(self, x):
        with torch.no_grad():
            pred = int(torch.argmax(self(x)).item())
            predName = self.taskMap[pred]
        return (predName, predName)

    def trainStep(self, x, n_x, name):
        o = self(x)
        ind = self.taskMap.index(name)
        y = torch.tensor(ind).unsqueeze(0)
        with torch.no_grad():
            pred = int(torch.argmax(o).item())
        self.opt.zero_grad()
        loss = self.lossFunc(o, y)
        loss.backward()
        self.opt.step()
        return (None, {"ce": loss.item(), "acc": (pred == ind)})

    def trainDistro(self, x, name):
        pass   # No distro maintained in this algo.

    def toggleSaving(self, on):
        self.saving = on

    def addTask(self, name, init = None):
        if len(self.taskMap) >= self.maxTasks:
            raise RuntimeError("Error: Classifier has a maximum number of tasks %s." % self.maxTasks)
        else:
            self.taskMap.append(name)

    def saveAll(self):
        self.saveDetector()

    def loadAll(self, names = []):
        for i, name in enumerate(names):
            self.addTask(name)
            filepath = os.path.join(self.savePath, "Baseline_classifier.pt")
            self.load_state_dict(torch.load(filepath, map_location = torch.device(self.dev)))

    def expelDetector(self, name = ""):
        self.saveDetector(name)

    def saveDetector(self, name = ""):
        if self.saving:
            filepath = os.path.join(self.savePath, "Baseline_classifier.pt")
            torch.save(self.state_dict(), filepath)










#===============================================================================
