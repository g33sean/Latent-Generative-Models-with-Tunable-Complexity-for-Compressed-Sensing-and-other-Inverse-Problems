# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py

from collections import namedtuple
import inspect
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as tvm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        features = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[features[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[features[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[i] for i in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[i] for i in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[i] for i in range(23, 30)])
        self.N_slices = 5
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        VggOutputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return VggOutputs(h1, h2, h3, h4, h5)


class LPIPS(nn.Module):
    def __init__(self, net='vgg', version='0.1', use_dropout=True):
        super().__init__()
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.L = len(self.chns)
        self.net = vgg16(requires_grad=False)
        self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in self.chns])

        model_path = os.path.abspath(
            os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth' % (version, net)))
        self.load_state_dict(torch.load(model_path, map_location=device), strict=False)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, in0, in1, normalize=False):
        if normalize:
            in0, in1 = 2 * in0 - 1, 2 * in1 - 1
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk] = nn.functional.normalize(outs0[kk], dim=1)
            feats1[kk] = nn.functional.normalize(outs1[kk], dim=1)
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        return sum(res)


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """1x1 conv layer for learned perceptual weighting."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
