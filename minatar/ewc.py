
'''
Algo adapted from https://github.com/moskomule/ewc.pytorch.
'''
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def _variable(t, device, **kwargs):
    t = t.to(device)
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, imp = 1.0, device = "cpu"):
        self.model = None
        self.dataset = None
        self.params = None
        self.means = None
        self.matrices = None
        self.imp = imp
        self.device = device

    def resetImportance(self, imp = 1.0):
        self.imp = imp

    def archive(self, model, oldSamples):
        self.model = model
        self.dataset = oldSamples
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.means = dict()
        self.matrices = self._diagFisherMatrix()
        for n, p in deepcopy(self.params).items():
            self.means[n] = _variable(p.data, self.device)

    def _diagFisherMatrix(self):
        matrices = dict()
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            matrices[n] = _variable(p.data, self.device)
        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            print("dogo", input.size())
            input = _variable(input, self.device)
            print("bogo", input.size())
            output = self.model(input).view(1, -1)
            print("mahogo", output.size())
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()
            for n, p in self.model.named_parameters():
                matrices[n].data += p.grad.data ** 2 / len(self.dataset)
        matrices = {n: p for n, p in matrices.items()}
        return matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            ploss = self.matrices[n] * (p - self.means[n]) ** 2
            loss += ploss.sum()
        return loss * self.imp





#===============================================================================
