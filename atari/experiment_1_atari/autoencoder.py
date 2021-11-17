import torch
import torch.nn as nn

KAMSE_WEIGHT = 10

class AutoEncoder(nn.Module):
    def __init__(self, inShape, h_dim, latent_size):
        super(AutoEncoder, self).__init__()
        self.zShape = latent_size

        self.inShape = inShape
        self.e_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3,stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(16),
            nn.Conv2d(16, 32,kernel_size=3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(32)
        )

        self.e_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128,kernel_size=3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1),
        )

        self.e_fc1 = nn.Linear(inShape[0] * inShape[1]*128, h_dim)
        self.e_fc2 = nn.Linear(h_dim, latent_size)

        self.d_fc1 = nn.Linear(latent_size, h_dim)
        self.d_fc2 = nn.Linear(h_dim, inShape[0] * inShape[1]*128)

        self.d_layer1 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.InstanceNorm2d(64),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.InstanceNorm2d(32)
        )

        self.d_layer2 = nn.Sequential(
        nn.ConvTranspose2d(32, 16, kernel_size=3,stride = 1, padding = 1),
        nn.LeakyReLU(0.1),
        nn.InstanceNorm2d(16),
        nn.ConvTranspose2d(16, 16, kernel_size=3, stride = 1, padding = 1),
        nn.LeakyReLU(),
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh())

        self.criterian = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def kamse_loss(self, y, y_2, y_hat):
        mse = nn.MSELoss()(y, y_hat)

        mask = y_2-y

        mask[mask != 0] = 1

        y_hat_masked = mask * y_hat
        y_2_masked = mask * y_2
        y_masked = mask * y

        k_loss = nn.MSELoss()(y_masked, y_hat_masked)

        return mse + KAMSE_WEIGHT * k_loss


    def encoder(self, x):
        x = self.e_layer1(x)
        x = self.e_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.e_fc1(x)
        x = self.e_fc2(x)
        return x

    def decoder(self, x):
        x = self.d_fc1(x)
        x = self.d_fc2(x)
        x = x.view(x.size(0), 128, self.inShape[0], self.inShape[1])
        x = self.d_layer1(x)
        x = self.d_layer2(x)
        return x

    def detect(self, x):
        with torch.no_grad():
            out = self.forward(x)
            return -self.criterian(out, x)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_step(self, x, n_x):
        out = self.forward(x)
        loss = self.kamse_loss(x, n_x, out)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return out, loss

    def generate(self):
        noise = torch.empty(self.zShape).normal_(mean = 0.0, std = 1.0).unsqueeze(0)
        g = self.decoder(noise)
        return g
