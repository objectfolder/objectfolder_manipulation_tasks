from torchvision.models import resnet18
import torch
import argparse
from torch import nn
from torch.nn.modules.activation import MultiheadAttention

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        vision_extractor = resnet18(pretrained=True)
        # flat all features to rgb channel
        if in_ch == 202:
            second_extractor = resnet18(pretrained=True)
            vision_extractor.conv1 = nn.Conv2d(
                112, 64, kernel_size=7, stride=1, padding=3, bias=False
            )
            second_extractor.conv1 = nn.Conv2d(
                90, 64, kernel_size=7, stride=1, padding=3, bias=False
            )
            modules_1 = list(vision_extractor.children())[:-1]
            self.encoder_1 = nn.Sequential(*modules_1)
            modules_2 = list(second_extractor.children())[:-1]
            self.encoder_2 = nn.Sequential(*modules_2)
        else:
            vision_extractor.conv1 = nn.Conv2d(
                in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False
            )
            # Remove the last fc layer, and rebuild
            modules = list(vision_extractor.children())[:-1]
            self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        if self.in_ch == 202:
            feat_1 = self.encoder_1(x[:,90:,...]).squeeze(-1).squeeze(-1)
            feat_2 = self.encoder_2(x[:,:90,...]).squeeze(-1).squeeze(-1)
            feat = torch.cat((feat_1, feat_2), dim=-1)
        else:
            feat = self.encoder(x).squeeze(-1).squeeze(-1)    
        return feat


class DynamicModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args == 202:
            self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1026, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )
        else:
            self.mlp = torch.nn.Sequential(
            torch.nn.Linear(514, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )
        
        self.mha = MultiheadAttention(512, 8)

    def forward(self, x, init, action):
        # feat plus init pose of obj and action(theta and dist)
        # feat = torch.cat((x, init, action), dim=-1)
        if self.args == 202:
            # x (batch, 1024)
            inp = torch.stack((x[:, :512], x[:, 512:]), dim=0) #(2, batch, 512)
            out, _ = self.mha(inp, inp, inp)
            out += inp
            feat = torch.concat([out[i] for i in range(out.shape[0])], 1)
        feat = torch.cat((x, action), dim=-1)
        pred = self.mlp(feat)
        return pred
    
class ForwardModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.dyn = DynamicModel(args)
        
    def forward(self, x_1, x_2, x_3, init, action):
        feat_1 = self.encoder(x_1)
        feat_2 = self.encoder(x_2)
        feat_3 = self.encoder(x_3)
        
        feat = (feat_1 + feat_2 + feat_3) / 3
        # feat = torch.zeros_like(feat)
        return self.dyn(feat, init, action)