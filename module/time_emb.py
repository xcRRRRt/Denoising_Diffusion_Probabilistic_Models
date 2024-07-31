import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, T=1000, d=100):
        super(TimeEmbedding, self).__init__()
        self.T = T
        self.dim = d

        self.time_embedding = nn.Embedding(T, d)
        self.time_embedding.weight.data = self._init_time_embedding()
        self.time_embedding.requires_grad = False

    def _init_time_embedding(self):
        embed = torch.zeros(self.T, self.dim)
        w = torch.Tensor([1 / 10000 ** (2 * j / self.dim) for j in range(self.dim)])
        w = w.reshape((1, self.dim))
        pos = torch.arange(self.T).reshape((self.T, 1)).float()
        embed[:, ::2] = torch.sin(pos * w[:, ::2])
        embed[:, 1::2] = torch.cos(pos * w[:, ::2])
        return embed

    def forward(self, t):
        return self.time_embedding(t)


if __name__ == '__main__':
    TimeEmbedding()