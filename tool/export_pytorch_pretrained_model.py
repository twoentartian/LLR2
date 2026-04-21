import argparse
import os
import ssl
import sys
from collections.abc import Callable

import torch
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_src.model_opti_save_load import save_model_state
from py_src.ml_setup_dataset import DatasetType
from py_src.ml_setup_model import ModelType
import py_src.third_party.compact_transformers.src.cct as cct


def _save_export(
    output_path: str,
    file_name: str,
    model: torch.nn.Module,
    model_type: ModelType,
    dataset_type: DatasetType,
) -> None:
    target_path = os.path.join(output_path, file_name)
    save_model_state(
        target_path,
        model.state_dict(),
        model_type.name,
        dataset_type.name,
    )
    print(f"[info] saved {target_path}")


def _build_export_jobs() -> list[tuple[str, Callable[[], torch.nn.Module], ModelType, DatasetType]]:
    imagenet1k = DatasetType.imagenet1k
    return [
        ("resnet18_imagenet1k.model.pt", lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), ModelType.resnet18_bn, imagenet1k),
        ("resnet34_imagenet1k.model.pt", lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1), ModelType.resnet34, imagenet1k),
        ("resnet50_imagenet1k_v1.model.pt", lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1), ModelType.resnet50, imagenet1k),
        ("resnet50_imagenet1k_v2.model.pt", lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), ModelType.resnet50, imagenet1k),
        ("vgg11_bn_imagenet1k_v1.model.pt", lambda: models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1), ModelType.vgg11_bn, imagenet1k),
        ("squeezenet1_1.model.pt", lambda: models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1), ModelType.squeezenet1_1, imagenet1k),
        ("shufflenet_v2_x2_0.model.pt", lambda: models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1), ModelType.shufflenet_v2_x2_0, imagenet1k),
        ("mobilenet_v3_large_imagenet_v1.model.pt", lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1), ModelType.mobilenet_v3_large, imagenet1k),
        ("mobilenet_v3_large_imagenet_v2.model.pt", lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2), ModelType.mobilenet_v3_large, imagenet1k),
        ("efficientnet_v2_s.model.pt", lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1), ModelType.efficientnet_v2_s, imagenet1k),
        ("efficientnet_b1_imagenet_v1.model.pt", lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1), ModelType.efficientnet_b1, imagenet1k),
        ("efficientnet_b1_imagenet_v2.model.pt", lambda: models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2), ModelType.efficientnet_b1, imagenet1k),
        ("mnasnet1_0_imagenet_v1.model.pt", lambda: models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1), ModelType.mnasnet1_0, imagenet1k),
        ("mnasnet0_5_imagenet_v1.model.pt", lambda: models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1), ModelType.mnasnet0_5, imagenet1k),
        ("densenet121_imagenet_v1.model.pt", lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1), ModelType.densenet121, imagenet1k),
        ("regnet_y_400_mf_imagenet_v1.model.pt", lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1), ModelType.regnet_y_400mf, imagenet1k),
        ("regnet_y_400_mf_imagenet_v2.model.pt", lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V2), ModelType.regnet_y_400mf, imagenet1k),
        ("convnext_tiny_imagenet_v1.model.pt", lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1), ModelType.convnext_tiny, imagenet1k),
        ("alexnet_imagenet.model.pt", lambda: models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1), ModelType.alexnet, imagenet1k),
        ("resnext50_32x4d_imagenet_v1.model.pt", lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1), ModelType.resnext50_32x4d, imagenet1k),
        ("resnext50_32x4d_imagenet_v2.model.pt", lambda: models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2), ModelType.resnext50_32x4d, imagenet1k),
        ("vit_b_32_imagenet_v1.model.pt", lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1), ModelType.vit_b_32, imagenet1k),
        ("wide_resnet50_2_imagenet_v1.model.pt", lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1), ModelType.wide_resnet50_2, imagenet1k),
        ("wide_resnet50_2_imagenet_v2.model.pt", lambda: models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2), ModelType.wide_resnet50_2, imagenet1k),
        ("cct_14_7x2_224.model.pt", lambda: cct.cct_14_7x2_224(pretrained=True, progress=False), ModelType.cct_14_7x2_224, imagenet1k),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch pretrained models to LLR2 .model.pt files")
    parser.add_argument(
        "-o",
        "--output_folder_name",
        default="pytorch_pretrained",
        help="output folder for exported checkpoints",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip models whose output files already exist",
    )
    args = parser.parse_args()

    output_path = args.output_folder_name
    os.makedirs(output_path, exist_ok=True)

    ssl._create_default_https_context = ssl._create_unverified_context

    export_jobs = _build_export_jobs()
    for file_name, builder, model_type, dataset_type in export_jobs:
        target_path = os.path.join(output_path, file_name)
        if args.skip_existing and os.path.exists(target_path):
            print(f"[info] skip existing {target_path}")
            continue

        print(f"[info] exporting {file_name}")
        model = builder()
        _save_export(output_path, file_name, model, model_type, dataset_type)


if __name__ == "__main__":
    main()
