import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, ch) -> None:
        super(Attention, self).__init__()
        self.linear_1 = nn.Linear(ch, ch)
        self.linear_2 = nn.Linear(ch, ch)
        self.linear_3 = nn.Linear(ch, ch)

        self.linear_final = nn.Linear(ch, ch)

    def forward(self, x):
        b, c, h, w = x.shape

        xt = x.view(b, c, h * w).transpose(1, 2)  # [b, h*w, c]
        key = self.linear_1(xt)
        value = self.linear_2(xt)
        query = self.linear_3(xt)
        query = query.view(b, -1, 1, c).transpose(1, 2)  # [b, 1, h*w, c]
        key = key.view(b, -1, 1, c).transpose(1, 2)
        value = value.view(b, -1, 1, c).transpose(1, 2)

        a = nn.functional.scaled_dot_product_attention(query, key, value)
        a = a.transpose(1, 2).reshape(b, -1, c)
        a = self.linear_final(a)
        a = a.transpose(-1, -2).reshape(b, c, h, w)

        return a + x


if __name__ == '__main__':
    in_ = torch.randn(4, 32, 16, 16)
    att = Attention(32)
    out_ = att(in_)
    print(out_.shape)
