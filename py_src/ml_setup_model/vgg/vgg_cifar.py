import torch.nn as nn

_VGG_CFG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


# ---------------------------------------------------------------------------
# VGG CIFAR (custom CIFAR-optimised VGG, ported from kuangliu/pytorch-cifar)
# ---------------------------------------------------------------------------

class VGGCifar(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super().__init__()
        self.features = self._make_layers(_VGG_CFG[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    @staticmethod
    def _make_layers(cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)
