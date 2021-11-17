import os
import torch
import json
import statistics
import time
from collections import defaultdict, deque



class TaskDetector:
    def __init__(self, anomalyDetectorGen, savePath, distroMemSize = 100, detectorCache = -1):
        super().__init__()
        self.gen = anomalyDetectorGen
        self.taskMap = dict()
        self.savePath = savePath
        self.distroMemSize = distroMemSize
        self.distroMap = defaultdict(self._buildDistro)
        self.cache = set()
        self.cacheCap = detectorCache
        self.cacheOn = (self.cacheCap >= 0)

    def detect(self, state):
        bestScore = float("-inf")
        task = None
        bestScoreWithNorm = float("-inf")
        taskWithNorm = None
        self._checkDetectorCache()
        with torch.no_grad():
            for name in self.taskMap.keys():
                distro = self._recalcDistro(name)
                model = self.getDetectorModel(name)
                model.eval()
                score = model.detect(state).item()
                normScore = (score - distro[0]) / distro[1]   # Normalize.
                if bestScore < score:
                    task = name
                    bestScore = score
                if bestScoreWithNorm < normScore:
                    taskWithNorm = name
                    bestScoreWithNorm = normScore
                if self.cacheOn and name not in self.cache:
                    self.expelDetector(name)
        return (task, taskWithNorm)

    def to(self, device):
        for name in self.taskMap.keys():
            model = self.getDetectorModel(name)
            model.to(device)

    def addTask(self, name, init = True):
        if init:
            self.taskMap[name] = self.gen.generateDetector()
        else:
            self.taskMap[name] = str(name)

    def getDetectorModel(self, name):
        model = self.taskMap[name]
        if isinstance(model, str):
            model = self.rebuildDetector(name)
        return model

    def expelDetector(self, name):
        self.saveDetector(name)
        self.taskMap[name] = str(name)

    def rebuildDetector(self, name):
        model = self.gen.generateDetector()
        filepath = os.path.join(self.savePath, "Det-%s.pt" % name)
        print("Rebuilding detector %s." % name)
        model.load_state_dict(torch.load(filepath, map_location = torch.device('cpu')))
        self.taskMap[name] = model
        return model

    def saveDetector(self, name):
        model = self.taskMap[name]
        if not isinstance(model, str):
            filepath = os.path.join(self.savePath, "Det-%s.pt" % name)
            print("Saving detector %s." % name)
            torch.save(model.state_dict(), filepath)

    def saveDistros(self):
        for name in self.taskMap.keys():
            self._recalcDistro(name)
        ob = json.dumps(self._dequesToLists(dict(self.distroMap)))
        filepath = os.path.join(self.savePath, "Distros.json")
        with open(filepath, "w") as outfile:
            outfile.write(ob)

    def loadDistros(self):
        filepath = os.path.join(self.savePath, "Distros.json")
        if os.path.isfile(filepath):
            with open(filepath) as jsonFile:
                d = self._listsToDeques(json.load(jsonFile))
        else:
            d = dict()
        self.distroMap = defaultdict(self._buildDistro, d)

    def saveAll(self):
        for name in self.taskMap.keys():
            self.saveDetector(name)
        self.saveDistros()

    def loadAll(self, names = []):
        if len(names) == 0:
            for name in self.taskMap.keys():
                self.rebuildDetector(name)
        else:
            for name in names:
                self.rebuildDetector(name)
        self.loadDistros()

    def trainStep(self, x, name):
        model = self.getDetectorModel(name)
        model.train()
        return model.train_step(x)

    def trainDistro(self, x, name):
        distro = self.distroMap[name]
        model = self.getDetectorModel(name)
        model.eval()
        score = model.detect(x).item()
        distro[2].append(score)
        distro[3] = True

    def resetCache(self, detectorCache = None):
        self.cache = set()
        if detectorCache is None:
            self.cacheCap = self.cacheCap
        else:
            self.cacheCap = detectorCache
            self.cacheOn = (self.cacheCap >= 0)
        self._checkDetectorCache()

    def resetDetectorDevice(self, name, dev):
        model = self.getDetectorModel(name)
        model.to(dev)
        if self.cacheOn and name not in self.cache:
            self.expelDetector(name)

    def _checkDetectorCache(self):
        for name in self.taskMap.keys():
            if len(self.cache) < self.cacheCap:
                self.cache.add(name)
            else:
                break

    def _recalcDistro(self, name):
        d = self.distroMap[name]
        m, sd, dMem, needsUpdate = d
        if needsUpdate:
            d = [statistics.mean(dMem), statistics.stdev(dMem), dMem, False]
            self.distroMap[name] = d
        return d

    def _buildDistro(self):
        return [0, 1, deque([], maxlen = self.distroMemSize), False]

    def _dequesToLists(self, distros):
        for k, distro in distros.items():
            distro[2] = list(distro[2])
        return distros

    def _listsToDeques(self, distros):
        for k, distro in distros.items():
            distro[2] = deque(distro[2], maxlen = self.distroMemSize)
        return distros







#===============================================================================
