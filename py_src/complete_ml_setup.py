"""Training hyperparameter presets - optimizers, LR schedulers, and epoch counts.

Ported from DFL_torch/py_src/complete_ml_setup.py and adapted for LLR2's MLSetup.
"""

import math
import torch

from py_src.ml_setup.ml_setup import MLSetup
from py_src.ml_setup_model import ModelType
from py_src.ml_setup_dataset import DatasetType


class FastTrainingSetup:
    @staticmethod
    def get_optimizer_lr_scheduler_epoch(
        arg_ml_setup: MLSetup,
        model,
        preset: int = 0,
        override_dataset=None,
        override_batch_size=None,
    ):
        err = NotImplementedError(
            f"no preset for {arg_ml_setup.model_type} @ {arg_ml_setup.dataset_type} preset={preset}"
        )

        training_data = arg_ml_setup.training_data if override_dataset is None else override_dataset
        batch_size = arg_ml_setup.default_batch_size if override_batch_size is None else override_batch_size
        steps_per_epoch = len(training_data) // batch_size + 1

        mt = arg_ml_setup.model_type
        dt = arg_ml_setup.dataset_type

        if mt in (ModelType.lenet5, ModelType.lenet4):
            if dt == DatasetType.mnist:
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
                return optimizer, None, 20
            elif dt == DatasetType.random_mnist:
                lr, epochs = 0.01, 100
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.lenet5_large_fc:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            return optimizer, None, 20

        elif mt in (ModelType.resnet18_bn, ModelType.resnet18_gn):
            if dt in (DatasetType.cifar10,):
                lr, epochs = 0.1, 70
                if preset == 0:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                elif preset == 1:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr * 2, momentum=0.9, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 2, steps_per_epoch=steps_per_epoch, epochs=epochs)
                elif preset == 2:
                    epochs = 100
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                else:
                    raise err
                return optimizer, scheduler, epochs
            elif dt == DatasetType.cifar100:
                lr, epochs = 0.1, 70
                if preset == 0:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                elif preset == 1:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr * 2, momentum=0.9, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr * 2, steps_per_epoch=steps_per_epoch, epochs=epochs)
                else:
                    raise err
                return optimizer, scheduler, epochs
            elif dt == DatasetType.svhn:
                lr, epochs = 0.1, 100
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                return optimizer, scheduler, epochs
            elif dt in (DatasetType.imagenet10, DatasetType.imagenet100, DatasetType.imagenet1k,
                        DatasetType.imagenet1k_sam_mask_random_noise, DatasetType.imagenet1k_sam_mask_black):
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-3,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.resnet34:
            if dt in (DatasetType.imagenet1k,):
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-3,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.resnet50:
            if dt in (DatasetType.imagenet1k, DatasetType.imagenet100):
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-3,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.simplenet:
            lr, epochs = 0.1, 150
            optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-3, weight_decay=0.001)
            milestones = [steps_per_epoch * i for i in [100, 190, 306, 390, 440, 540]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
            return optimizer, scheduler, epochs

        elif mt == ModelType.vgg11_bn:
            if dt == DatasetType.cifar10:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.vgg11_no_bn:
            if dt == DatasetType.cifar10:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.cct_7_3x1_32:
            if dt in (DatasetType.cifar10,):
                if preset == 0:
                    weight_decay, initial_lr = 6e-2, 55e-5
                elif preset == 1:
                    weight_decay, initial_lr = 1e-2, 100e-5
                else:
                    raise err
            elif dt == DatasetType.cifar100:
                weight_decay, initial_lr = 6e-2, 6e-4
            else:
                raise err
            warmup_lr, min_lr = 1e-5, 1e-5
            warmup_epochs, epochs, cooldown_epochs = 10, 300, 10
            warmup_steps = warmup_epochs * steps_per_epoch
            cosine_steps = (epochs - warmup_epochs) * steps_per_epoch
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
            def lr_lambda(step):
                if step < warmup_steps:
                    lr = warmup_lr + (initial_lr - warmup_lr) * (step / warmup_steps)
                elif step < warmup_steps + cosine_steps:
                    t = step - warmup_steps
                    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * t / cosine_steps))
                else:
                    lr = min_lr
                return lr / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return optimizer, scheduler, epochs
        
        elif mt == ModelType.cct_14_7x2_224:
            if dt in (DatasetType.imagenet1k,):
                weight_decay, initial_lr = 5e-2, 5e-4
            else:
                raise err
            warmup_lr, min_lr = 0.000001, 0.00001
            warmup_epochs, epochs, cooldown_epochs = 25, 300, 10
            warmup_steps = warmup_epochs * steps_per_epoch
            cosine_steps = (epochs - warmup_epochs) * steps_per_epoch
            optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
            def lr_lambda(step):
                if step < warmup_steps:
                    lr = warmup_lr + (initial_lr - warmup_lr) * (step / warmup_steps)
                elif step < warmup_steps + cosine_steps:
                    t = step - warmup_steps
                    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * t / cosine_steps))
                else:
                    lr = min_lr
                return lr / initial_lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return optimizer, scheduler, epochs

        elif mt == ModelType.mobilenet_v2:
            if dt == DatasetType.cifar10:
                epochs = 200
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=4e-5, momentum=0.9)
                milestones = [steps_per_epoch * 100]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
            elif dt == DatasetType.cifar100:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=4e-5, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
            else:
                raise err
            return optimizer, scheduler, epochs

        elif mt == ModelType.efficientnet_b0:
            if dt in (DatasetType.cifar10, DatasetType.cifar100):
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.shufflenet_v2:
            if dt == DatasetType.cifar10:
                epochs = 300
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=4e-5, momentum=0.9)
                milestones = [steps_per_epoch * i for i in [150, 225]]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
            elif dt == DatasetType.cifar100:
                epochs = 200
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=4e-5, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
            else:
                raise err
            return optimizer, scheduler, epochs

        elif mt in (ModelType.densenet121, ModelType.densenet_cifar):
            if dt == DatasetType.cifar10:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.regnet_x_200mf:
            if dt in (DatasetType.cifar10, DatasetType.cifar100):
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.dla_46_c:
            if dt == DatasetType.imagenet10:
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.dla:
            if dt in (DatasetType.cifar10, DatasetType.cifar100):
                epochs = 120
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.efficientnet_v2_s:
            if dt == DatasetType.imagenet1k:
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-3,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.efficientnet_b1:
            if dt == DatasetType.imagenet1k:
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-3,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.mobilenet_v3_large:
            if dt == DatasetType.imagenet1k:
                epochs = 100
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-4, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs * steps_per_epoch, eta_min=1e-3,
                )
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.ddpm_cifar10:
            if dt == DatasetType.cifar10:
                epochs = 512
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
                return optimizer, None, epochs
            else:
                raise err

        elif mt == ModelType.nanoclip_default:
            if dt == DatasetType.flickr30k:
                epochs, warmup_epochs, lr = 400, 5, 1e-3
                weight_decay = 4e-4
                optimizer_params = [
                    {"params": model.img_encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
                    {"params": model.txt_encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
                ]
                optimizer = torch.optim.AdamW(optimizer_params)
                total_steps = epochs * steps_per_epoch
                warmup_steps = warmup_epochs * steps_per_epoch
                warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
                return optimizer, scheduler, epochs
            else:
                raise err

        elif mt == ModelType.transformer_for_grokking:
            if dt in (DatasetType.arithmetic_addition, DatasetType.arithmetic_cubepoly, DatasetType.arithmetic_cube2):
                total_steps, warmup_steps = 150000, 10
                optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0, lr=1e-3, betas=(0.9, 0.98), eps=1e-8)
                warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-4)
                scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
                return optimizer, scheduler, total_steps
            else:
                raise err

        else:
            raise err


class TransferTrainingSetup:
    @staticmethod
    def get_optimizer_lr_scheduler_epoch(src_dataset_type: DatasetType, arg_ml_setup: MLSetup, model, preset: int = 0):
        err = NotImplementedError(
            f"no transfer preset for {arg_ml_setup.model_type} @ {arg_ml_setup.dataset_type} from {src_dataset_type}"
        )
        mt, dt = arg_ml_setup.model_type, arg_ml_setup.dataset_type
        steps_per_epoch = len(arg_ml_setup.training_data) // arg_ml_setup.default_batch_size + 1

        if mt == ModelType.resnet18_bn:
            if dt == DatasetType.svhn and src_dataset_type == DatasetType.cifar10:
                lr, epochs = 0.01, 30
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
                return optimizer, scheduler, epochs
            else:
                raise err
        else:
            raise err
