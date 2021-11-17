
import itertools
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
            assert False, "Temporary stop."


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True




class Encoder(nn.Module):
    def __init__(self, inShape, zShape, h = 300):
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
        self.fcZ = nn.Linear(h, zShape)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        z = self.fcZ(x)
        return z



class Decoder(nn.Module):
    def __init__(self, inShape, imgShape, h = 300):
        super().__init__()
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
        nn.LeakyReLU(0.1),
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, self.imgShape[0], self.imgShape[1])
        x = self.deconvBlock1(x)
        x = self.deconvBlock2(x)
        return x


'''
class Descriminator(nn.Module):
    def __init__(self, zShape, h1 = 300, h2 = 100):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(zShape, h1),
            nn.LeakyReLU(0.1),
            nn.Dropout(p = 0.05),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.1),
            nn.Dropout(p = 0.05),
            nn.Linear(h2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)
'''


class Descriminator(nn.Module):
    def __init__(self, h1 = 300, h2 = 100):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(32),
            nn.Flatten(),
            nn.Linear(16000, h1),
            nn.LeakyReLU(0.1),
            #nn.Dropout(p = 0.05),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.1),
            #nn.Dropout(p = 0.05),
            nn.Linear(h2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)




class AAE(nn.Module):
    def __init__(self, enc, dec, des, zShape):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.descriminator = des
        self.zShape = zShape
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.criterian = self.mse
        self.optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=1e-5, weight_decay=1e-5)
        self.descOptimizer = torch.optim.Adam(self.descriminator.parameters(), lr=1e-5, weight_decay=1e-5)
        self.encOptimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5, weight_decay=1e-5)

    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return (o, z)

    def detect(self, x):
        with torch.no_grad():
            o, z = self.forward(x)
            return -self.criterian(o, x)
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
    def trainReconst(self, x, n_x):
        o, z = self.forward(x)
        loss = self.mse(n_x, o)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (o, z, loss.item())

    def trainDesc(self, x):
        #encoderComp = self.encoder.state_dict()
        targReal = torch.tensor([[1.0]])
        targFake = torch.tensor([[0.0]])
        o, z = self.forward(x)
        og = torch.randn_like(o)
        po = self.descriminator(o)
        loss1 = self.bce(po, targFake)
        self.descOptimizer.zero_grad()
        loss1.backward()
        self.descOptimizer.step()
        #print("DESC-R:   ", po.tolist(), "   ", targReal.tolist())
        pg = self.descriminator(og)
        loss2 = self.bce(pg, targReal)
        self.descOptimizer.zero_grad()
        loss2.backward()
        self.descOptimizer.step()
        #print("DESC-F:   ", pg.tolist(), "   ", targFake.tolist())
        #if not compare_state_dict(encoderComp, self.encoder.state_dict()):
            #print("Error: Unknown enc change.")
            #print(encoderComp)
            #print(self.encoder.state_dict())
            #raise RuntimeError
        return (loss1.item() + loss2.item()) * 0.5

    def trainReg(self, x):
        #descComp = self.descriminator.state_dict()
        o, z = self.forward(x)
        p = self.descriminator(o)
        targ = torch.tensor([[1.0]])
        #print("REG:   ", p.tolist(), "   ", targ.tolist())
        loss = self.bce(p, targ)
        self.encOptimizer.zero_grad()
        loss.backward()
        self.encOptimizer.step()
        #if not compare_state_dict(descComp, self.descriminator.state_dict()):
            #print("Error: Unknown desc change.")
            #print(descComp)
            #print(self.descriminator.state_dict())
            #raise RuntimeError
        return loss.item()
    '''
    def trainDesc(self, x):
        targReal = torch.tensor([[1.0]])
        targFake = torch.tensor([[0.0]])
        zx = self.encoder(x)
        zg = torch.randn_like(zx)
        px = self.descriminator(zx)
        loss1 = self.bce(px, targFake)
        self.descOptimizer.zero_grad()
        loss1.backward()
        self.descOptimizer.step()
        print("DESC-R:   ", px.tolist(), "   ", targReal.tolist())
        pg = self.descriminator(zg)
        loss2 = self.bce(pg, targReal)
        self.descOptimizer.zero_grad()
        loss2.backward()
        self.descOptimizer.step()
        print("DESC-F:   ", pg.tolist(), "   ", targFake.tolist())
        return (loss1.item() + loss2.item()) * 0.5

    def trainReg(self, x):
        zx = self.encoder(x)
        p = self.descriminator(zx)
        targ = torch.tensor([[1.0]])
        print("REG:   ", p.tolist(), "   ", targ.tolist())
        loss = self.bce(p, targ)
        self.encOptimizer.zero_grad()
        loss.backward()
        self.encOptimizer.step()
        return loss.item()
    '''
    def train_step(self, x, n_x):
        o, z, recL = self.trainReconst(x.clone().detach(), n_x.clone().detach())
        descL = 0.0
        runs = 1
        for i in range(runs):
            descL += self.trainDesc(x.clone().detach())
        descL = descL / runs
        regL = 0.0
        runs = 1
        for i in range(runs):
            regL += self.trainReg(x.clone().detach())
        regL = regL / runs
        return (o, z, {"reconst_loss": recL, "descrim_loss": descL, "regular_loss": regL})

    def generate(self):
        noise = torch.empty(self.zShape).normal_(mean = 0.0, std = 1.0).unsqueeze(0)
        g = self.decoder(noise)
        return g

    def getEncoder(self):
        return self.encoder

    def getDecoder(self):
        return self.decoder

    def getDescriminator(self):
        return self.descriminator






#===============================================================================
