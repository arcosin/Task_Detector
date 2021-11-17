

import os
import json
import torch
import statistics
from collections import defaultdict, deque

from .autoencoder import AutoEncoder
'''
class AnomalyDetectorGenerator:
    def __init__(self, device, inShape, h_dim, latent_size):
        super().__init__()
        self.device = device
        self.inShape = inShape
        self.h_dim = h_dim
        self.latent_size = latent_size

    def generateDetector(self):
        return AutoEncoder(self.inShape, self.h_dim, self.latent_size).to(self.device)
'''



class TaskDetector:
    def __init__(self, anomalyDetectorGen, savePath, detType, distroMemSize = 100, detectorCache = -1, device = "cpu"):
        super().__init__()
        self.gen = anomalyDetectorGen
        self.taskMap = dict()
        self.savePath = savePath
        self.saving = True
        self.distroMemSize = distroMemSize
        self.distroMap = defaultdict(self._buildDistro)
        self.cache = set()
        self.cacheCap = detectorCache
        self.cacheOn = (self.cacheCap >= 0)
        self.detType = detType
        self.dev = device

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
        if self.saving:
            self.saveDetector(name)
        self.taskMap[name] = str(name)

    def rebuildDetector(self, name):
        model = self.gen.generateDetector()
        filepath = os.path.join(self.savePath, "Det-%s-%s.pt" % (self.detType, name))
        #print("Rebuilding detector %s." % name)
        model.load_state_dict(torch.load(filepath, map_location = torch.device(self.dev)))
        #print("Done loading.")
        self.taskMap[name] = model
        return model

    def saveDetector(self, name):
        model = self.taskMap[name]
        if not isinstance(model, str):
            filepath = os.path.join(self.savePath, "Det-%s-%s.pt" % (self.detType, name))
            print("Saving detector %s." % name)
            torch.save(model.state_dict(), filepath)

    def saveDistros(self):
        for name in self.taskMap.keys():
            self._recalcDistro(name)
        ob = json.dumps(dict(self.distroMap))
        filepath = os.path.join(self.savePath, "Distros-%s.json" % self.detType)
        with open(filepath, "w") as outfile:
            outfile.write(ob)

    def loadDistros(self):
        filepath = os.path.join(self.savePath, "Distros-%s.json" % self.detType)
        if os.path.isfile(filepath):
            d = json.loads(filepath)
        else:
            print("No distro save found.")
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
            for i, name in enumerate(names):
                if i < self.cacheCap:
                    self.rebuildDetector(name)
                else:
                    self.taskMap[name] = ""
        self.loadDistros()

    def trainStep(self, x, n_x, name):
        model = self.getDetectorModel(name)
        model.train()
        return model.train_step(x, n_x)

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

    def toggleSaving(self, on):
        self.saving = on

    def getNames(self):
        return list(self.taskMap.keys())

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







#===============================================================================
