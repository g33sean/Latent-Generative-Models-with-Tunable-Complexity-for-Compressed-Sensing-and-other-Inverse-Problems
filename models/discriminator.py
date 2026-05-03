import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=5,
                 norm='instance'):
        super().__init__()
        ch = base_channels
        norm_layer = {
            'batch':    lambda c: nn.BatchNorm2d(c, affine=True),
            'instance': lambda c: nn.InstanceNorm2d(c, affine=True),
            'group':    lambda c: nn.GroupNorm(32, c)
        }[norm]

        layers = [
            spectral_norm(nn.Conv2d(in_channels, ch, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            stride = 2 if n < n_layers - 1 else 1
            layers += [
                spectral_norm(nn.Conv2d(ch * nf_mult_prev,
                                        ch * nf_mult, 4, stride, 1, bias=False)),
                norm_layer(ch * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        layers += [spectral_norm(nn.Conv2d(ch * nf_mult, 1, 4, 1, 1, bias=False))]
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')

    def forward(self, x):
        return self.model(x)
