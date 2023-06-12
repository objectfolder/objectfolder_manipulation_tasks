import torch
import torch.nn as nn


class VGG_layer(nn.Module):
    def __init__(self, nin, nout):
        super(VGG_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True) # The official pytorch implementation uses ReLU here, but SVG uses LeakyReLU
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1, nout=128, expand=1):
        super(encoder, self).__init__()
        self.dim = dim
        # 64 x 64
        self.c1 = nn.Sequential(
                VGG_layer(nc, int(64*expand)),
                VGG_layer(int(64*expand), int(64*expand)),
                )
        # 32 x 32
        self.c2 = nn.Sequential(
                VGG_layer(int(64*expand), int(128*expand)),
                VGG_layer(int(128*expand), int(128*expand)),
                )
        # 16 x 16
        self.c3 = nn.Sequential(
                VGG_layer(int(128*expand), int(256*expand)),
                VGG_layer(int(256*expand), int(256*expand)),
                VGG_layer(int(256*expand), int(256*expand)),
                )

        self.c4 = nn.Conv2d(int(256*expand), 128, 3, 1, 1) # One conv layer with 128 outputs
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) # 64 -> 32
        h2 = self.c2(self.mp(h1)) # 32 -> 16
        h3 = self.c3(self.mp(h2)) # 16 -> 8
        h4 = self.c4(self.mp(h3)) # 8 -> 4
        return h4, [h1, h2, h3]


class decoder(nn.Module):
    def __init__(self, dim, nc=1, expand=1):
        super(decoder, self).__init__()
        self.dim = dim
        # 16 x 16

        self.upc1 = nn.Conv2d(128, int(256*expand), 3, 1, 1) # One conv layer with 128 outputs

        self.upc2 = nn.Sequential(
                VGG_layer(int(256*2*expand), int(256*expand)),
                VGG_layer(int(256*expand), int(256*expand)),
                VGG_layer(int(256*expand), int(128*expand))
                )
        # 32 x 32
        self.upc3 = nn.Sequential(
                VGG_layer(int(128*2*expand), int(128*expand)),
                VGG_layer(int(128*expand), int(64*expand))
                )
        # 64 x 64
        self.upc4 = nn.Sequential(
                VGG_layer(int(64*2*expand), int(64*expand)),
                nn.ConvTranspose2d(int(64*expand), nc, 3, 1, 1),
                nn.Sigmoid()
                )

        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input):
        vec, skip = input
        up2 = self.up(self.upc1(vec))
        d3 = self.upc2(torch.cat([up2, skip[2]], 1)) # 16 x 16
        up3 = self.up(d3) # 8 -> 32
        d4 = self.upc3(torch.cat([up3, skip[1]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        output = self.upc4(torch.cat([up4, skip[0]], 1)) # 64 x 64
        return output
