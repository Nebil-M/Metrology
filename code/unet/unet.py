from itertools import pairwise

import torch
import torch.nn as nn


def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode="replicate"),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode="replicate"),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), *double_conv(in_c, out_c))

    def forward(self, x):
        return self.pool_conv(x)


class Decoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = double_conv(2 * out_c, out_c)

    def forward(self, x, prev):
        x = self.up(x)
        x = torch.concat([x, prev], dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, channels):
        super().__init__()

        self.first = double_conv(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        for in_c, out_c in pairwise(channels):
            self.encoders.append(Encoder(in_c, out_c))

        self.decoders = nn.ModuleList()
        for in_c, out_c in pairwise(channels[::-1]):
            self.decoders.append(Decoder(in_c, out_c))

        self.final = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.first(x)

        convs = []
        for enc in self.encoders:
            convs.append(x)
            x = enc(x)

        convs.reverse()

        for i, dec in enumerate(self.decoders):
            x = dec(x, convs[i])

        x = self.final(x)
        return x
