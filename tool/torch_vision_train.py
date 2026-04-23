import datetime
import os
import sys
import time
import warnings

import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from py_src import model_variance_correct
from py_src import special_torch_layers
from py_src.ml_setup_dataset.dataset_default import imagenet1k_path
from py_src.model_opti_save_load import load_model_state_file
from py_src.torch_vision_train import get_mixup_cutmix
from py_src.torch_vision_train.sampler import RASampler
import py_src.torch_vision_train.presets as presets
import py_src.torch_vision_train.utils as utils


def _append_eval_log(output_dir: str | None, message: str) -> None:
    if output_dir is None or not utils.is_main_process():
        return
    test_info_path = os.path.join(output_dir, "test.txt")
    with open(test_info_path, "a", encoding="utf-8") as file_handle:
        file_handle.write(message + "\n")


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    variance_correction = args.variance_correction
    variance_correction_on_norm = args.variance_correction_on_norm
    print(f"variance correction is {variance_correction}.")
    if not hasattr(train_one_epoch, "batch_norm_layer_names"):
        norm_layers = special_torch_layers.find_normalization_layers(model)
        batch_norm_layer_names, _ = special_torch_layers.find_layers_according_to_name_and_keyword(
            model.state_dict(), [], norm_layers.batch_normalization
        )
        print(f"totally {len(batch_norm_layer_names)} batch-normalization layers: {batch_norm_layer_names}.")
        layer_norm_layer_names, _ = special_torch_layers.find_layers_according_to_name_and_keyword(
            model.state_dict(), [], norm_layers.layer_normalization
        )
        print(f"totally {len(layer_norm_layer_names)} layer-normalization layers: {layer_norm_layer_names}.")
        assert len(norm_layers.group_normalization) == 0, "group normalization layers are not supported yet."
        assert len(norm_layers.instance_normalization) == 0, "instance normalization layers are not supported yet."

        train_one_epoch.batch_norm_layer_names = batch_norm_layer_names
        train_one_epoch.layer_norm_layer_names = layer_norm_layer_names
        train_one_epoch.ignore_move_layer_names = batch_norm_layer_names + layer_norm_layer_names
        train_one_epoch.all_norm_layer_names = batch_norm_layer_names + layer_norm_layer_names
        train_one_epoch.variance_correction_norm_layer_names = []
        if variance_correction and variance_correction_on_norm:
            print("Batch norm layers are included for variance correction.")
            train_one_epoch.variance_correction_norm_layer_names += batch_norm_layer_names
        if not args.silence:
            input("Please check above information and press Enter to continue, or press Ctrl+C to quit")

    target_variance = None
    if variance_correction:
        variance_record = model_variance_correct.VarianceCorrector(
            model_variance_correct.VarianceCorrectionType.FollowOthers
        )
        variance_record.add_variance(model.state_dict())
        target_variance = variance_record.get_variance()

    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if variance_correction and target_variance is not None:
            ignore_layer_list = [] if variance_correction_on_norm else train_one_epoch.all_norm_layer_names
            target_model_state_dict = model_variance_correct.VarianceCorrector.scale_model_stat_to_variance(
                model.state_dict(), target_variance, ignore_layer_list=ignore_layer_list
            )
            model.load_state_dict(target_model_state_dict)

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / max(time.time() - start_time, 1e-12))


def evaluate(model, criterion, data_loader, device, output_dir=None, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    total_processed = int(num_processed_samples.item())
    if hasattr(data_loader.dataset, "__len__") and len(data_loader.dataset) != total_processed and utils.is_main_process():
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {total_processed} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    summary = f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    if utils.is_main_process():
        print(summary)
    _append_eval_log(output_dir, summary)
    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    digest = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", digest[:10] + ".pt")
    return os.path.expanduser(cache_path)


def load_data(traindir, valdir, args):
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    start_time = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - start_time)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])
        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        if args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    utils.init_distributed_mode(args)
    if args.output_dir:
        utils.mkdir(args.output_dir)
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "args.txt"), "w", encoding="utf-8") as file_handle:
                file_handle.write(" ".join(sys.argv))

    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    if args.load_existing_weights is not None:
        model_state, _, _ = load_model_state_file(args.load_existing_weights)
        print(f"loading existing model weights from {args.load_existing_weights}")
        model.load_state_dict(model_state)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if custom_keys_weight_decay else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema and "model_ema" in checkpoint:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, output_dir=args.output_dir, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device, output_dir=args.output_dir)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device, output_dir=args.output_dir)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, output_dir=args.output_dir, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    time_now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    default_data_path = str(imagenet1k_path)
    default_output_folder_name = f"{os.path.basename(__file__)}_{time_now_str}"
    default_output_folder_path = os.path.join(os.curdir, default_output_folder_name)

    parser.add_argument("--data-path", default=default_data_path, type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (use cuda or cpu)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, total batch size is world_size x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for normalization layers (defaults to --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters (defaults to --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for transformer embedding parameters (defaults to --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="learning-rate scheduler")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="number of warmup epochs")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="warmup method")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="warmup decay")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by this factor")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum learning rate")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=default_output_folder_path, type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="cache datasets for quicker initialization; also serializes transforms",
        action="store_true",
    )
    parser.add_argument("--sync-bn", dest="sync_bn", help="use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="only test the model", action="store_true")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of RandAugment")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of AugMix")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability")
    parser.add_argument("--amp", action="store_true", help="use torch.cuda.amp mixed precision")

    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="how often to update the EMA model",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="EMA decay factor",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="force deterministic algorithms only"
    )
    parser.add_argument("--interpolation", default="bilinear", type=str, help="interpolation method")
    parser.add_argument("--val-resize-size", default=256, type=int, help="validation resize size")
    parser.add_argument("--val-crop-size", default=224, type=int, help="validation crop size")
    parser.add_argument("--train-crop-size", default=224, type=int, help="training crop size")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="maximum gradient norm")
    parser.add_argument("--ra-sampler", action="store_true", help="use repeated augmentation sampler")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repeated-augmentation repetitions")
    parser.add_argument("--weights", default=None, type=str, help="torchvision weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor")
    parser.add_argument("--use-v2", action="store_true", help="use torchvision v2 transforms")
    parser.add_argument("--load-existing-weights", default=None, type=str, help="load existing model weights from path")
    parser.add_argument("--variance-correction", action="store_true", help="enable variance correction")
    parser.add_argument(
        "--variance-correction-on-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include normalization layers in variance correction",
    )
    parser.add_argument(
        "--disable-distributed",
        "--disable_distributed",
        dest="disable_distributed",
        action="store_true",
        help="disable distributed training",
    )
    parser.add_argument(
        "-S",
        "--silence",
        action="store_true",
        help="silence interactive prompts and bypass manual checks",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.data_path = os.path.expanduser(args.data_path)
    args.output_dir = os.path.expanduser(args.output_dir)
    main(args)
