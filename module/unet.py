import torch
import torch.nn as nn

from module.time_emb import TimeEmbedding


class Block(nn.Module):
    def __init__(self, shape, in_channels, out_channels, in_dim):
        super(Block, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LayerNorm(shape),
            nn.SiLU(),
        )
        self.part2 = _make_time_embedding_layer(in_dim, out_channels)
        self.part3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LayerNorm(shape),
            nn.SiLU(),
        )

        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.SiLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.SiLU(inplace=True)
        # )

    def forward(self, x, t):
        p1 = self.part1(x)
        p2 = self.part2(t).reshape(x.shape[0], -1, 1, 1)
        p3 = self.part3(p1 + p2)
        return p3
        # return self.double_conv(x)


def _make_time_embedding_layer(in_, out):
    return nn.Sequential(
        nn.Linear(in_, out),
        nn.SiLU(inplace=True),
        nn.Linear(out, out),
    )


class UNet(nn.Module):
    def __init__(self, in_feature: int, num_features: int, multi: list[int], shape=(1, 32, 32), ):
        super(UNet, self).__init__()
        channels = [in_feature] + [num_features * m for m in multi]
        self.channels = channels

        self.shapes = []
        c, h, w = shape
        for i in range(len(multi)):
            self.shapes.append((channels[i + 1], h, w))
            h //= 2
            w //= 2

        # self.down_blocks = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.down_pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(len(channels) - 2)])

        # self.up_blocks = nn.ModuleList([Block(channels[i + 1], channels[i]) for i in range(len(channels) - 2, -1, -1)])
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
            self.down_blocks = nn.ModuleList([Block(self.shapes[i], self.channels[i], self.channels[i + 1], value.dim) for i in range(len(self.channels) - 1)])
            self.up_blocks = nn.ModuleList([Block(self.shapes[i - 1], self.channels[i + 1], self.channels[i], value.dim) for i in range(len(self.channels) - 2, 0, -1)])

            # self.embedding_layers_down = nn.ModuleList([_make_time_embedding_layer(value.dim, self.channels[i]) for i in range(len(self.channels) - 1)])
            # self.embedding_layers_up = nn.ModuleList([_make_time_embedding_layer(value.dim, self.channels[i] * 2) for i in range(len(self.channels) - 2, -1, -1)])


if __name__ == '__main__':
    input_ = torch.randn(4, 3, 64, 64)
    model = UNet(3, 64, [1, 2, 4, 8], shape=(3, 64, 64))
    setattr(model, "time_embedding", TimeEmbedding(1000, 100))
    # print(model)
    print(model(input_, torch.randint(0, model.time_embedding.T, (4,))).shape)

# class MyBlock(nn.Module):
#     def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
#         super(MyBlock, self).__init__()
#         self.ln = nn.LayerNorm(shape)
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
#         self.activation = nn.SiLU() if activation is None else activation
#         self.normalize = normalize
#
#     def forward(self, x):
#         out = self.ln(x) if self.normalize else x
#         out = self.conv1(out)
#         out = self.activation(out)
#         out = self.conv2(out)
#         out = self.activation(out)
#         return out
#
#
# class UNet(nn.Module):
#     def __init__(self, n_steps=1000, time_emb_dim=100):
#         super(UNet, self).__init__()
#
#         # Sinusoidal embedding
#         self.time_embed = TimeEmbedding(n_steps, time_emb_dim)
#         self.time_embed.requires_grad_(False)
#
#         # First half
#         self.te1 = self._make_te(time_emb_dim, 1)
#         self.b1 = nn.Sequential(
#             MyBlock((1, 28, 28), 1, 10),
#             MyBlock((10, 28, 28), 10, 10),
#             MyBlock((10, 28, 28), 10, 10)
#         )
#         self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
#
#         self.te2 = self._make_te(time_emb_dim, 10)
#         self.b2 = nn.Sequential(
#             MyBlock((10, 14, 14), 10, 20),
#             MyBlock((20, 14, 14), 20, 20),
#             MyBlock((20, 14, 14), 20, 20)
#         )
#         self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
#
#         self.te3 = self._make_te(time_emb_dim, 20)
#         self.b3 = nn.Sequential(
#             MyBlock((20, 7, 7), 20, 40),
#             MyBlock((40, 7, 7), 40, 40),
#             MyBlock((40, 7, 7), 40, 40)
#         )
#         self.down3 = nn.Sequential(
#             nn.Conv2d(40, 40, 2, 1),
#             nn.SiLU(),
#             nn.Conv2d(40, 40, 4, 2, 1)
#         )
#
#         # Bottleneck
#         self.te_mid = self._make_te(time_emb_dim, 40)
#         self.b_mid = nn.Sequential(
#             MyBlock((40, 3, 3), 40, 20),
#             MyBlock((20, 3, 3), 20, 20),
#             MyBlock((20, 3, 3), 20, 40)
#         )
#
#         # Second half
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(40, 40, 4, 2, 1),
#             nn.SiLU(),
#             nn.ConvTranspose2d(40, 40, 2, 1)
#         )
#
#         self.te4 = self._make_te(time_emb_dim, 80)
#         self.b4 = nn.Sequential(
#             MyBlock((80, 7, 7), 80, 40),
#             MyBlock((40, 7, 7), 40, 20),
#             MyBlock((20, 7, 7), 20, 20)
#         )
#
#         self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
#         self.te5 = self._make_te(time_emb_dim, 40)
#         self.b5 = nn.Sequential(
#             MyBlock((40, 14, 14), 40, 20),
#             MyBlock((20, 14, 14), 20, 10),
#             MyBlock((10, 14, 14), 10, 10)
#         )
#
#         self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
#         self.te_out = self._make_te(time_emb_dim, 20)
#         self.b_out = nn.Sequential(
#             MyBlock((20, 28, 28), 20, 10),
#             MyBlock((10, 28, 28), 10, 10),
#             MyBlock((10, 28, 28), 10, 10, normalize=False)
#         )
#
#         self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)
#
#     def forward(self, x, t):
#         # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
#         t = self.time_embed(t)
#         n = len(x)
#         out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
#         out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
#         out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)
#
#         out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)
#
#         out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
#         out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)
#
#         out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
#         out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)
#
#         out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
#         out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)
#
#         out = self.conv_out(out)
#
#         return out
#
#     def _make_te(self, dim_in, dim_out):
#         return nn.Sequential(
#             nn.Linear(dim_in, dim_out),
#             nn.SiLU(),
#             nn.Linear(dim_out, dim_out)
#         )
