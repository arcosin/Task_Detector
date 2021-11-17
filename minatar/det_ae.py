

import torch
import torch.nn as nn

Z_DIM = 32



class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cl = nn.Conv2d(10, 8, kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(512, Z_DIM)
        self.fc2 = nn.Linear(Z_DIM, 800)
        self.ctl = nn.ConvTranspose2d(8, 10, kernel_size = 3, stride = 1, padding = 1)
        self.criterian = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def encoder(self, x):
        x = self.cl(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

    def decoder(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 8, 10, 10)
        x = self.ctl(x)
        return x

    def detect(self, x):
        with torch.no_grad():
            out = self.forward(x)
            return -self.criterian(out, x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_step(self, x):
        out = self.forward(x)
        loss = self.criterian(out, x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return out, loss.item()











#===============================================================================
