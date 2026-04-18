import torch.nn as nn

def timm_build_loaders(cfg: dict, data_root: str):
    from timm.data import create_dataset, create_loader # type: ignore
    img_size = int(cfg["img_size"])
    input_size = (3, img_size, img_size)

    # ImageNet expects folder structure like:
    #   <data_root>/train/<class>/*.JPEG
    #   <data_root>/val/<class>/*.JPEG   (or "validation" depending on your setup)
    train_ds = create_dataset(
        name=cfg["dataset"],
        root=data_root,
        split="train",
        is_training=True,
        batch_size=cfg["batch_size"],
    )
    val_ds = create_dataset(
        name=cfg["dataset"],
        root=data_root,
        split="val",  # sometimes "validation"
        is_training=False,
        batch_size=cfg["batch_size"],
    )

    common = dict(
        input_size=input_size,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        crop_pct=float(cfg["crop_pct"]),
        pin_memory=True,
        use_prefetcher=False,  # <- set True only if you want timm to move/normalize on-GPU
    )

    train_loader = create_loader(
        train_ds, # type: ignore
        is_training=True,
        # map your YAML keys to timm create_loader args
        scale=tuple(cfg["scale"]),
        interpolation=str(cfg["train_interpolation"]),  # "random"
        auto_augment=str(cfg["aa"]),                    # YAML "aa" -> timm "auto_augment"
        re_prob=float(cfg["reprob"]),
        re_mode=str(cfg["remode"]),
        **common, # type: ignore
    )

    val_loader = create_loader(
        val_ds, # type: ignore
        is_training=False,
        interpolation=str(cfg["interpolation"]),        # "bicubic"
        **common, # type: ignore
    )

    return train_loader, val_loader

def timm_build_mixup_and_loss(cfg: dict):
    from timm.data import Mixup # type: ignore
    from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy # type: ignore
    use_mix = (float(cfg.get("mixup", 0)) > 0) or (float(cfg.get("cutmix", 0)) > 0)
    use_mix = False
    mixup_fn = None
    if use_mix:
        mixup_fn = Mixup(
            mixup_alpha=float(cfg["mixup"]),
            cutmix_alpha=float(cfg["cutmix"]),
            prob=float(cfg["mixup_prob"]),
            switch_prob=float(cfg["mixup_switch_prob"]),
            mode=str(cfg["mixup_mode"]),  # "batch"
            label_smoothing=float(cfg["smoothing"]),
            num_classes=int(cfg["num_classes"]),
        )
        criterion = SoftTargetCrossEntropy()
    else:
        s = float(cfg["smoothing"])
        criterion = LabelSmoothingCrossEntropy(smoothing=s) if s > 0 else nn.CrossEntropyLoss()

    return mixup_fn, criterion