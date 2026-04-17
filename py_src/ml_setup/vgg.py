import torch.nn as nn

from py_src.adapters import StandardAdapter
from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType

from py_src.ml_setup_model.vgg import VGG11_no_bn
from py_src.ml_setup_dataset import dataset_cifar10

def _make_setup(model, model_type, dataset_setup, batch_size, has_normalization=True, criterion=None, clip_grad_norm=None) -> MLSetup:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    adapter = StandardAdapter(model, criterion, clip_grad_norm=clip_grad_norm)
    return MLSetup(
        model=model,
        adapter=adapter,
        model_type=model_type,
        training_data=dataset_setup.train_data,
        testing_data=dataset_setup.valdation_data,
        dataset_type=dataset_setup.dataset_type,
        default_batch_size=batch_size,
        has_normalization_layer=has_normalization,
    )

_VGG_CFG = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


# ---------------------------------------------------------------------------
# VGG CIFAR (custom CIFAR-optimised VGG, ported from kuangliu/pytorch-cifar)
# ---------------------------------------------------------------------------

class _VGGCifar(nn.Module):
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


# ---------------------------------------------------------------------------
# VGG
# ---------------------------------------------------------------------------

def vgg11_bn_cifar10() -> MLSetup:
    model = _VGGCifar('VGG11', num_classes=10)
    return _make_setup(model, ModelType.vgg11_bn, dataset_cifar10(), 256)


def vgg11_no_bn_cifar10() -> MLSetup:
    model = VGG11_no_bn(in_channels=3, num_classes=10)
    return _make_setup(model, ModelType.vgg11_no_bn, dataset_cifar10(rescale_to_224=True), 32, has_normalization=False)
