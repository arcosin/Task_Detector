

import torch
import torch.nn as nn


KAMSE_WEIGHT = 10


def checkBad(label, vars, names):
    for name, var in zip(names, vars):
        if torch.isnan(var).any() or torch.isinf(var).any():
            print("\n\n", label, "\n\n")
            print(name)
            print("Max", torch.max(var))
            print("Min", torch.min(var))
            print("Avg", torch.mean(var))
            print("Nans", torch.isnan(var).any())
            print("Infs", torch.isinf(var).any())
            print()
            for n, v in zip(names, vars):
                if name != n:
                    print(n)
                    print("Max", torch.max(v))
                    print("Min", torch.min(v))
                    print("Avg", torch.mean(v))
                    print("Nans", torch.isnan(v).any())
                    print("Infs", torch.isinf(v).any())
                    print()
            print("\n\n")
            raise RuntimeError





class Encoder(nn.Module):
    def __init__(self, inShape, muSize, sigSize, h = 300):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(32)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
        )
        self.fc = nn.Linear(inShape[0] * inShape[1] * 128, h)
        self.fcMu = nn.Linear(h, muSize)
        self.fcSig = nn.Linear(h, sigSize)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu = self.fcMu(x)
        sig = self.fcSig(x)
        return (mu, sig)



class Decoder(nn.Module):
    def __init__(self, inShape, imgShape, h = 300):
        super().__init__()
        self.zShape = inShape
        self.imgShape = imgShape
        self.fc1 = nn.Linear(inShape, h)
        self.fc2 = nn.Linear(h, imgShape[0] * imgShape[1] * 128)
        self.deconvBlock1 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.InstanceNorm2d(64),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.InstanceNorm2d(32)
        )
        self.deconvBlock2 = nn.Sequential(
        nn.ConvTranspose2d(32, 16, kernel_size=3,stride = 1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.InstanceNorm2d(16),
        nn.ConvTranspose2d(16, 16, kernel_size=3, stride = 1, padding = 1),
        nn.LeakyReLU(),
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, self.imgShape[0], self.imgShape[1])
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        return x



class VAE(nn.Module):
    def __init__(self, enc, dec, reconstWeight = 1.0, divergeWeight = 1e-6):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.reconLoss = nn.MSELoss()
        self.rw = reconstWeight
        self.dw = divergeWeight
        self.criterian = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def forward(self, x):
        mu, sig = self.encoder(x)
        lv = self.reparamaterize(mu, sig)
        o = self.decoder(lv)
        return (o, mu, sig, lv)

    def detect(self, x):
        with torch.no_grad():
            o, mu, sig, lv = self.forward(x)
            return -self.criterian(o, x)

    def loss(self, data, reconst, mu, sig):
        rl = self.reconLoss(reconst, data)
        compMeans = torch.full(sig.size(), 0.0)
        compStd = torch.full(sig.size(), 1.0)
        dl = self.kld(mu, sig, compMeans, compStd)
        totalLoss = self.rw * rl + self.dw * dl
        checkBad("loss", [rl, dl, totalLoss], ["reconst", "kl-div", "total"])
        return (totalLoss, rl, dl)
        '''
    def kamse_loss(self, y, y_2, y_hat):
        mse = nn.MSELoss()(y, y_hat)
        mask = y_2-y
        mask[mask != 0] = 1
        y_hat_masked = mask * y_hat
        y_2_masked = mask * y_2
        y_masked = mask * y
        k_loss = nn.MSELoss()(y_masked, y_hat_masked)
        return mse + KAMSE_WEIGHT * k_loss
        '''
    def kld(self, mu1, std1, mu2, std2):
        p = torch.distributions.Normal(mu1, std1)
        q = torch.distributions.Normal(mu2, std2)
        kl = torch.distributions.kl_divergence(p, q).mean()
        checkBad("kld-end", [mu1, mu2, kl], ["mu1", "mu2", "kl"])
        return kl

    def reparamaterize(self, mu, sig):
        epsMeans = torch.full(sig.size(), 0.0)
        epsStd = torch.full(sig.size(), 1.0)
        eps = torch.normal(epsMeans, epsStd)   #torch.randn_like(sig)
        checkBad("reparam", [sig, mu, eps, eps * sig + mu], ["sig", "mu", "eps", "sample"])
        return eps * sig + mu

    def train_step(self, x, n_x):
        o, mu, sig, lv = self.forward(x)
        loss, rl, dl = self.loss(n_x, o, mu, sig)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (o, {"loss": loss.item(), "reconst": rl.item(), "diverge": dl.item()})

    def generate(self):
        zDim = self.decoder.zShape
        noise = torch.empty(zDim).normal_(mean = 0.0, std = 1.0).unsqueeze(0)
        g = self.decoder(noise)
        return g

    def getEncoder(self):
        return self.encoder

    def getDecoder(self):
        return self.decoder

    def scaleDivLoss(self, mul):
        self.dw *= mul

    def scaleRecLoss(self, mul):
        self.rw *= mul






#===============================================================================
