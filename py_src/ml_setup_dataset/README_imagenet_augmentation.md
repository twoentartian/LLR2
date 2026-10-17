# ImageNet 增强：v1/v2 与 v3（旧 ViT 配方）

核对日期：2026-10-17。本文记录 **LLR2 实际实现的增强行为**，用于配置实验、解释历史结果和后续 LLM 分析。
图片增强入口位于同目录的 [dataset_imagenet.py](dataset_imagenet.py)，batch 增强和 sampler 位于
[imagenet_preset.py](../ml_setup/imagenet_preset.py)，ViT 的选择逻辑位于 [vit.py](../ml_setup/vit.py)。

**历史结论：修复前 ViT 普通训练的 preset 0 和 1 都启用了强增强，包括 Mixup/CutMix。
这套配方现正式命名为 v3，新代码通过 `preset=3` 选择。
历史日志中的 0/1 仍是原始编号，不能标记成项目 ImageNet v2，也不能把旧 preset 0 标成无增强。**

## 1. 版本和适用范围

- 历史证据：提交 `7484fdc`（`prepare for surf snellius experiments.`）中的 `py_src/ml_setup/vit.py`。
  该提交同时包含 ImageNet-1k 和 ImageNet-21k-p 的旧 ViT 配方。
- 提交 `6b26901` 的 ViT/ImageNet-1k 入口也使用该旧配方。更早的实验应查询实际运行提交，不能仅凭本文推断。
- 修复后的 ViT 代码用 `VIT_PRESET_VERSION = 2` 标识 preset 语义；普通训练的 `info.json`
  和 `training_checkpoint.pt` 的 `run_config` 记录 `vit_preset_version=2`。
- `VIT_LEGACY_PRESET = 3` 标识历史兼容配方 v3。语义版本仍为 2，因为当前 0/1/2 的含义保持不变；
  `vit_preset_version=2, preset=3` 正是合法的 v3 配置。
- 涉及四种模型：`vit_b_32`、`vit_b_32_reg4`、`vit_b_32_rope`、`vit_b_32_swiglu`，
  数据集为 `imagenet1k` 和 `imagenet21k_p`。
- 下表描述默认 dataset builder 和 collate/sampler 路径。显式 `override_dataset`、自定义 loader、
  单独实验脚本或修改过的工作树可能覆盖这些设置，分析时应读取实际入口。

本文的 **v1/v2 是项目的 ImageNet 配方编号**。它与 Python 的 `torchvision.transforms` /
`torchvision.transforms.v2` API、torchvision 模型权重的 `IMAGENET1K_V1/V2` 名称分别独立。
例如旧 ViT 的 `use_v2=True` 只是选择 transforms v2 API，并不能据此认定采用了项目 v2 配方。
项目配方的准确含义以本地代码为准，不应仅凭“官方 v1/v2”的简称推断上游参数。

## 2. v3：旧 ViT 的完整训练配方

在 `7484fdc` 中，ViT 仅接受 preset 0/1，preset 2 会报错；0 和 1 都进入同一套配置。
当前 `preset=3` 恢复这套完整配置，包括原 transforms API、batch collate、sampler 和损失。
默认图像处理流程如下：

```text
RandomResizedCrop(224, interpolation=BILINEAR)
→ RandomHorizontalFlip(p=0.5)
→ AutoAugment(policy=IMAGENET, interpolation=BILINEAR)
→ 转换成 float32 tensor，像素缩放到 [0, 1]
→ Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
→ batch 随机选择 MixUp(alpha=0.2) 或 CutMix(alpha=1.0)
```

此外还有以下设置：

- `RASampler(..., repetitions=3)`；DALI 路径的 `repeated_augmentation_repetitions=3`。
- 分类损失为 `CrossEntropyLoss(label_smoothing=0.11)`。
- `random_erase_prob=0.0`，因此没有 RandomErasing。
- 没有使用项目 v2 的 TrivialAugmentWide 和 RandAugment。
- Mixup/CutMix 的实现为 `RandomChoice([MixUp, CutMix])`，每个 batch 随机选一种，
  并非在每个 batch 上依次执行两种。启用后标签是类别概率向量；ImageNet-1k 为 1000 维，
  21k-p 的宽度跟随实际分类头和类别目录数。

旧普通训练的验证图像采用 `Resize(256) → CenterCrop(224) → tensor → Normalize`，
不使用训练的 AutoAugment、Mixup/CutMix 或重复增强。

标签平滑属于损失正则化，重复增强属于采样策略，Mixup/CutMix 属于 batch 增强；
它们均影响实验配方，不能只检查单张图片的 `dataset.transform` 就判断增强是否关闭。

## 3. 与当前 preset 0、项目 v1/v2 的对照

下表固定为 ViT 的 224×224 输入，以便只比较增强选项。

| 项目 | 当前 ViT preset 0 | 项目 v1 / 当前 ViT preset 1 | 项目 v2 / 当前 ViT preset 2 | v3 / 历史 ViT preset 0/1 |
|---|---|---|---|---|
| 训练裁剪 | Resize(256) + CenterCrop(224) | RandomResizedCrop(224) | RandomResizedCrop(224) | RandomResizedCrop(224) |
| 随机水平翻转 | 无 | p=0.5 | p=0.5 | p=0.5 |
| ImageNet AutoAugment | 无 | 无 | 无 | **有** |
| TrivialAugmentWide | 无 | 无 | **有** | 无 |
| RandAugment | 无 | 无 | **num_ops=2, magnitude=9** | 无 |
| RandomErasing | 无 | 默认无 | **p=0.1** | 无 |
| Mixup/CutMix | 无 | 无 | alpha=0.2 / 1.0，随机选一种 | alpha=0.2 / 1.0，随机选一种 |
| 重复增强参数 | 不启用 | 不启用 | **repetitions=4** | **repetitions=3** |
| CrossEntropy label smoothing | 0 | 0 | **0.1** | **0.11** |
| tensor 和 ImageNet 归一化 | 有 | 有 | 有 | 有 |
| 普通训练验证处理 | Resize(256) + CenterCrop(224) + 归一化 | 同左 | 同左 | 同左 |

旧 ViT 相比 v1，增加了 AutoAugment、Mixup/CutMix、重复增强和标签平滑。
旧 ViT 与 v2 共享 Mixup/CutMix，但单图策略、擦除、重复次数和标签平滑系数不同。
它们是两套不同配方，不能仅凭配置断言哪套一定更强、精度更高或训练更快。

**尺寸例外：** `get_imagenet_preprocessing(version=2)` 的通用默认值为训练裁剪 176、
验证 resize 232、验证裁剪 224；当前 ViT 为适配固定输入，显式覆盖为训练裁剪 224、
验证 resize 256、验证裁剪 224。不要直接把其他模型的通用 v2 尺寸套到 ViT。

**preset 0 的范围：** 无增强的 0 是 ViT 修复入口的明确定义。
通用 `imagenet_preset.preset_version()` 当前仍将 `preset != 1` 映射为 v2；
不能据此认为所有 ImageNet 模型的 preset 0 都已关闭增强，应查看各自的 builder。

## 4. 重复增强与 epoch 的实际训练量

设原始训练图片数为 `N`。以下忽略尾批次截断、DDP 补齐和显式子采样，
并以核对时的 sampler / DALI 实现为准。

| 配方 | PyTorch 普通单进程 sampler | PyTorch 分布式 RASampler（所有 rank 合计） | DALI（所有 rank 合计） |
|---|---|---|---|
| v3 / 旧 ViT，repetitions=3 | 约 `3N` / epoch | 约 `N` / epoch | 约 `3N` / epoch |
| 当前 ViT preset 0/1，无重复增强 | 约 `N` / epoch | 约 `N` / epoch | 约 `N` / epoch |
| 当前 ViT preset 2，repetitions=4 | 约 `4N` / epoch | 约 `N` / epoch | 约 `4N` / epoch |

`RASampler` 在非分布式路径重复全部 indices；分布式路径先创建重复序列、分片，再截取
`num_selected_samples`。大数据集上该值约为每卡 `N/world_size`，所以总训练量不直接乘以 repetitions。
它也不等同于每个 epoch 完整覆盖所有原始图片。

DALI 当前按 `N * repetitions` 设置样本预算，并遍历 reader 多次。
因此即便选择同一个 preset、同一个全局 batch、同样的 epochs，切换 PyTorch/DALI 后端仍可能改变
总更新次数。**相同 epoch 数不代表相同训练预算；比较实验和估算耗时时应核对
`len(train_loader)`、`steps_per_epoch`、batch size 和总 optimizer steps。**
这也是此前按旧 ViT DALI 的三倍训练量估时的原因；preset 1 没有三倍重复，
而 SURF 的提交脚本现默认 `preset=3`，恢复历史的三倍 DALI 训练量。

PyTorch 和 DALI 使用各自的随机裁剪、插值、增强和 RNG 实现，也不保证逐位一致。
当前共享 DALI pipeline 是 CPU JPEG 解码，之后在 GPU 做缩放、增强和归一化；
随机擦除的矩形采样/边界处理也与 torchvision 实现不同。保持配方选项一致不代表像素完全相同。

## 5. misalignment2 使用独立的 augmentation 参数

`tool/misalignment_measurement_2.py` 通过 `--augmentation none/0/1/2` 控制图像增强，
并使用 [augmentation.py](../../tool/misalignment/augmentation.py) 和自己的 loader。
它会移除原 dataset transform，再为 A/B 两个分区绑定同一套独立增强。
ViT/ImageNet 默认采用 MLSetup factory **preset 3**，以继承历史损失；其他 ImageNet 模型
仍默认 preset 1。可用 `--model_preset` 显式覆盖该选择。

- `none/0`：确定性 Resize / CenterCrop / Normalize。
- `1`：RandomResizedCrop + RandomHorizontalFlip + Normalize。
- ImageNet-1k 的 `2`：再加项目 v2 的 TrivialAugmentWide、RandAugment、RandomErasing；
  通用输入默认 176，固定 224 的 ViT 需要匹配输入尺寸，不能直接照搬。
- ImageNet-21k-p 只支持 `none/0` 和 `1`，`2` 明确报错。
- 不使用普通训练的 batch Mixup/CutMix，也不使用该 repeated-augmentation sampler；
  A/B 两侧采用相同增强策略，并在评估时继续使用配置的分区增强。
- 损失仍继承 `ml_setup.criterion`。ViT 默认的 `--model_preset 3` 恢复旧的 smoothing=0.11；
  `--model_preset 1` 则使用 smoothing=0。
  “没有使用 Mixup/CutMix”不能推导成“损失设置完全不受 ViT preset 修复影响”。

所以，不能从普通训练的旧 preset bug 推断“此前所有 misalignment 或其他项目都跑了 v2”。
应分别检查普通训练 `--preset` 和 misalignment 的 `--augmentation`、criterion、loader。

**misalignment 的 `--augmentation 3` 不代表 ViT v3**；ImageNet 不支持这个增强 level。
历史兼容组合例如 `--model_preset 3 --augmentation none` 或 `--model_preset 3 --augmentation 1`，
仍然不使用 Mixup/CutMix 或重复增强。结果记录 `model_preset` 和 `criterion.label_smoothing`。

普通训练通过 `--preset 3` 选择 v3；SURF 的 `tool/snellius/vit_b_32_imagenet21k_p.sbatch`
默认 `LLR2_PRESET=3`。直接调用 Python 脚本或 setup builder 时仍应显式传入 3，不能假设全项目
默认值都已改成 v3。v3 仅扩展 ViT 配方，不为其他模型自动引入新的 preset 3。

v3 恢复增强、采样和损失配置；历史比较还需对齐数据分区、后端、batch、总更新次数、
optimizer/scheduler、随机种子和精度。普通训练的新运行默认 BF16 评估；若历史指标来自 FP32，
可传 `--eval_precision fp32`，SURF 对应 `LLR2_EVAL_PRECISION=fp32`。
恢复相同配方不等于跨后端、跨 GPU 数或跨软件版本逐位复现。

## 6. 后续 LLM 如何识别和描述旧实验

1. 先定位入口脚本、实际运行 commit 和 dirty working-tree 记录。对历史代码用
   `git show <实际提交>:py_src/ml_setup/vit.py`，不要拿当前 preset 定义解释旧运行。
2. 普通训练读取 `info.json`、training checkpoint 的 `run_config` 以及命令行/日志，
   特别检查 `vit_preset_version`、`preset`、`dali`、`world_size`、batch 和 `steps_per_epoch`。
   旧版 `info.json` 不一定包含 preset，checkpoint/启动命令更有帮助。
3. `vit_preset_version=2` 表示本文的修复后 preset 语义（支持 0/1/2/3）；它不是“使用增强 preset 2”的标志。
   缺少该字段只能说明需要查证版本，单凭缺失不能认定实验采用了哪套增强。
4. 若确认代码为 `7484fdc` 的默认 ViT 普通训练路径且 preset 为 0/1，描述为：
   **“v3（旧 ViT）：RRC224 + HFlip + ImageNet AutoAugment + Mixup/CutMix(0.2/1.0)
   + RA3 + label smoothing 0.11”**，并补充实际后端/分布式的训练量。
5. 不要将其重命名成当前 preset 0、当前 preset 1 或项目标准 v2；不能只凭已保存的模型权重
   反推出训练增强。对证据不足的历史运行，应明确写“增强配方尚未确认”。
6. 普通训练兼容本项目的旧 ViT training checkpoint：对缺少 `vit_preset_version` 且保存 preset 0/1
   的旧格式，明确映射为 preset 3，日志报告迁移，新 checkpoint 的 `run_config` 记录原编号
   `vit_legacy_preset`。optimizer、scheduler、RNG、batch 和 world-size 的恢复规则保持原样。
   若旧 checkpoint 没有评估精度字段，恢复时默认保留历史 FP32 评估；显式 `--eval_precision` 可覆盖。
   已带语义版本 2 的 preset 0/1/2 不会被迁移。未知语义版本明确报错。
   如需改用其他配方，应以 `--initial_model` 开始新运行，不把恢复训练当作更换增强的入口。

## 7. 代码与验证入口

- [dataset_imagenet.py](dataset_imagenet.py)：`get_imagenet_preprocessing`、DALI pipeline、重复读取预算。
- [imagenet_preset.py](../ml_setup/imagenet_preset.py)：criterion、Mixup/CutMix、RASampler 参数。
- [vit.py](../ml_setup/vit.py)：ViT preset 0/1/2/3、224 输入覆盖、类别数。
- [presets.py](../torch_vision_train/presets.py)：旧 ViT 使用的 ClassificationPresetTrain / Eval。
- [transforms.py](../torch_vision_train/transforms.py)：`get_mixup_cutmix` 的 RandomChoice。
- [sampler.py](../torch_vision_train/sampler.py)：单进程与分布式 RASampler 的样本数。
- [misalignment_measurement_2.py](../../tool/misalignment_measurement_2.py)：独立分区、增强和 criterion。
- [test_vit_imagenet21k.py](../../test/test_vit_imagenet21k.py) 和
  [test_inference_precision.py](../../test/test_inference_precision.py)：当前 preset 的实际 loader / 标签 / 采样检查。
- [test_vit_v3.py](../../test/test_vit_v3.py)：直接加载 `7484fdc` 历史源码，在同一后端、固定种子下
  比较实际图像、混合标签、采样数量和初始权重；覆盖四种 ViT、1k/21k-p、PyTorch/DALI。

后续若修改配方、默认尺寸、采样预算或 checkpoint 语义，应同步更新本文和对应测试。
