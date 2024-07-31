import torch
from torch import nn

from module.time_emb import TimeEmbedding
from module.unet import UNet


class DDPM(nn.Module):
    def __init__(self, net: UNet, T=1000, d=100, beta=(1e-4, 0.02)):
        super(DDPM, self).__init__()
        self.net = net
        self.T = T
        self.beta = beta

        setattr(net, "time_embedding", TimeEmbedding(T, d))

        betas = torch.linspace(*beta, T)
        alphas = 1 - betas
        alpha_bars = torch.Tensor([torch.prod(alphas[: t + 1]) for t in range(len(alphas))])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(self, x_0):
        n, c, h, w = x_0.size()

        t = torch.randint(0, self.T, (x_0.shape[0],)).to(x_0.device)
        eps = torch.randn(n, c, h, w).to(x_0.device)

        alpha_bars = self.alpha_bars[t].reshape(n, -1, 1, 1)

        x_noisy = torch.sqrt(alpha_bars) * x_0 + torch.sqrt(1 - alpha_bars) * eps
        eps_theta = self.net(x_noisy, t)
        return x_noisy, eps, eps_theta

    def sample(self, sample_num: int, shape, device):
        with torch.no_grad():
            x = torch.randn(sample_num, *shape, device=device)
            for t in reversed(range(self.T)):
                time_tensor = torch.full((sample_num, 1), t, device=device, dtype=torch.long)
                eta_theta = self.net(x, time_tensor)
                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]
                x = (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta) / alpha_t.sqrt()
                if t > 0:
                    x += self.betas[t].sqrt() * torch.randn(sample_num, *shape, device=device)
        return x
