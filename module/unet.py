import torch
import torch.nn as nn

from module.time_emb import TimeEmbedding


class Block(nn.Module):
    def __init__(self, shape, in_channels, out_channels, in_dim, drop_out=0.1):
        super(Block, self).__init__()
        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LayerNorm(shape),
            nn.SiLU(),
        )
        self.part2 = _make_time_embedding_layer(in_dim, out_channels, drop_out)
        self.part3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LayerNorm(shape),
            nn.SiLU(),
        )

    def forward(self, x, t):
        p1 = self.part1(x)
        p2 = self.part2(t).reshape(x.shape[0], -1, 1, 1)
        p3 = self.part3(p1 + p2)
        return p3


def _make_time_embedding_layer(in_, out, drop_out=0.1):
    if drop_out > 0:
        return nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(in_, out),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(out, out)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_, out),
            nn.SiLU(inplace=True),
            nn.Linear(out, out),
        )


class UNet(nn.Module):
    def __init__(
            self,
            in_feature: int,
            num_features: int,
            multi: list[int],
            shape=(1, 32, 32),
            drop_out: float = 0.1
    ):
        super(UNet, self).__init__()
        assert 0 <= drop_out <= 1
        self.drop_out = drop_out
        channels = [in_feature] + [num_features * m for m in multi]
        self.channels = channels

        self.shapes = []
        c, h, w = shape
        for i in range(len(multi)):
            self.shapes.append((channels[i + 1], h, w))
            h //= 2
            w //= 2

        self.down_pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(len(channels) - 2)])

        self.up_transposes = nn.ModuleList([nn.ConvTranspose2d(channels[i + 1], channels[i], 2, 2) for i in range(len(channels) - 2, 0, -1)])

        self.final_conv = nn.Conv2d(num_features, in_feature, kernel_size=1)

    def forward(self, x, t):
        t = self.time_embedding(t)
        crops = []

        for down_block, down_pool in zip(self.down_blocks[:-1], self.down_pools):
            x = down_block(x, t)
            crops.append(x)
            x = down_pool(x)

        x = self.down_blocks[-1](x, t)
        crops = crops[::-1]

        for crop, up_transpose, up_block in zip(crops, self.up_transposes, self.up_blocks):
            x = up_transpose(x)
            x = torch.cat([x, crop], dim=1)
            x = up_block(x, t)

        x = self.final_conv(x)
        return x

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "time_embedding":
            value: TimeEmbedding
            self.down_blocks = nn.ModuleList([
                Block(self.shapes[i], self.channels[i], self.channels[i + 1], value.dim, self.drop_out)
                for i in range(len(self.channels) - 1)
            ])
            self.up_blocks = nn.ModuleList([
                Block(self.shapes[i - 1], self.channels[i + 1], self.channels[i], value.dim, self.drop_out)
                for i in range(len(self.channels) - 2, 0, -1)
            ])


if __name__ == '__main__':
    input_ = torch.randn(4, 3, 64, 64)
    model = UNet(3, 64, [1, 2, 4, 8], shape=(3, 64, 64))
    setattr(model, "time_embedding", TimeEmbedding(1000, 100))
    # print(model)
    print(model(input_, torch.randint(0, model.time_embedding.T, (4,))).shape)
