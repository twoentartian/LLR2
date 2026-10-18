# ImageNet 增强：v1/v2、v3（旧 ViT）与 v4（ConvNeXt）

核对日期：2026-10-18。本文记录 **LLR2 实际实现的增强行为**，用于配置实验、解释历史结果和后续 LLM 分析。
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

### 1.1 整套配方的来源与项目改动

下表区分**方法来源、上游参考实现、LLR2 实际组合**。引用某篇论文不意味着当前 preset
完整复现该论文；preset 编号是 LLR2 的接口。外部源码链接尽量固定到 tag / commit，
它们是本次核对的参考快照，不代表历史运行一定安装了该版本。

| LLR2 配方 | 可以核对的来源 | LLR2 保留或修改的部分 |
|---|---|---|
| preset 0（ViT / ConvNeXt-21k-p） | 项目定义的无随机增强基线；图像处理沿用 [torchvision 验证预处理][tv-presets] | 训练也执行 Resize / CenterCrop / Normalize；关闭混合标签、RA 和 smoothing。上游没有对应的统一“preset 0”。 |
| preset 1 / 项目 v1 | [torchvision 分类训练基础预处理][tv-presets]；本地 [98addde 的数据预处理文件][llr2-import] 自述移植自 DFL_torch | RRC + HFlip + Normalize，不启用 AA、TA、RandAugment、Mixup/CutMix、RA 或 smoothing。DFL_torch 的原始上游提交未在本地文件中记录，不能虚构精确移植版本。 |
| preset 2 / 项目 v2 | [torchvision 改进训练配方文章][tv-recipe] 提供 TA Wide、擦除 0.1、Mixup/CutMix 0.2/1.0、RA4、smoothing 0.1 等参考；实际组合见 [98addde][llr2-import] 及本地 `imagenet_preset.py` | **本项目额外将 RandAugment(2,9) 串接在 TrivialAugmentWide 后面**。官方参考脚本的 `auto_augment_policy` 是选择一种策略，并不这样串联。因此这是受官方配方启发的项目改版，不能称为官方 v2 的逐项复现。通用 176/232/224 尺寸来自该参考，ViT 显式覆盖为 224/256/224。 |
| preset 3 / v3（旧 ViT） | [torchvision 分类 README 的 ViT-B/32 命令][tv-vit]；本地 [c4de35f][llr2-vit-origin] 引入相应增强组合，[7484fdc][llr2-legacy-vit] 保存本项目历史版本 | AutoAugment(IMAGENET)、Mixup/CutMix 0.2/1.0、默认 RA3、smoothing **0.11** 都能在该上游参考命令 / 默认参数中找到依据。LLR2 的优化器、epoch 预算、EMA 使用和 DALI 适配另有变化，不能称为完整官方训练复现。 |
| preset 4（ConvNeXt-21k-p） | [ConvNeXt 论文附录 A.1 / 表 5][convnext-paper]、[官方 main.py][convnext-main] 的增强参数与 [datasets.py][convnext-datasets] 的构造方式 | 使用 timm RandAugment、Mixup/CutMix 0.8/1.0、像素擦除 0.25、标签平滑 0.1、bicubic，RA1。原始目标为 ImageNet-22k；当前使用 21k-p。CPU 调用当前 timm，DALI 是本项目的 GPU 适配，详见第 6 节。 |

**v3 的 0.11 不是随意改成的 0.1，也不是原始 ViT 论文的统一规定。**
它直接对应 torchvision 的 **ViT-B/32** 训练参考；不要套用 ViT-B/16 的 RandAugment 设置，
也不要把当前 90 epochs / 5 warmup / 本地 LR 当成该上游参考命令的训练预算。

### 1.2 各个增强操作、采样和损失项的来源

下表覆盖本文的普通训练 preset 0/1/2/3/4，以及 misalignment 的 ImageNet 增强。
RA 和标签平滑也列出，便于核对整个实验配方，但它们分别属于采样和损失正则化。

| 操作 / 设置 | 方法或实现来源 | 本项目如何使用 |
|---|---|---|
| RandomResizedCrop、水平翻转 | [torchvision 分类基础预处理][tv-presets]；ConvNeXt 使用 [timm transform factory][timm-transforms] | 通常 scale=(0.08,1.0)、ratio=(3/4,4/3)、HFlip p=0.5。DALI crop 解码器有自己的尝试 / 回退算法，不宣称与 PIL 逐像素相同。 |
| Resize / CenterCrop、tensor 转换、ImageNet mean/std | [torchvision 分类验证预处理][tv-presets] | 确定性预处理，也用于 preset 0；归一化与 resize 不意味着开启随机增强。插值和尺寸以 preset 为准。 |
| ImageNet AutoAugment | Cubuk 等，[AutoAugment][aa-paper]；[torchvision AutoAugment 策略表][tv-autoaugment]；[DALI AutoAugment 策略表][dali-aa] | v3 使用固定的已搜索策略，不在训练时重新搜索。**两个后端选择的策略表不同**，见下面的后端说明。 |
| TrivialAugmentWide | Müller、Hutter，[TrivialAugment][ta-paper]；[torchvision 实现][tv-autoaugment]、[DALI suite][dali-ta] | 项目 v2 使用 Wide 操作范围，并继续执行一次独立 RandAugment；后者是项目组合。 |
| RandAugment(2,9) | Cubuk 等，[RandAugment][ra-paper]；[torchvision 实现][tv-autoaugment]、[DALI suite][dali-ra] | 项目 v2 的 num_ops=2、magnitude=9；它不是 ConvNeXt 的 Gaussian / increasing 变体。 |
| `rand-m9-mstd0.5-inc1` | RandAugment 方法 + [timm `auto_augment.py`][timm-aa] 的具体变体；由 [ConvNeXt 默认参数][convnext-main] 选择 | preset 4 使用 increasing 操作池、2 次抽样、每操作 p=0.5、Gaussian 幅度 9±0.5。`mstd` / `inc1` 的具体含义来自 timm 实现，不能只引用 RandAugment 论文推断。 |
| Mixup | Zhang 等，[mixup: Beyond Empirical Risk Minimization][mixup-paper]；[torchvision reference helper][tv-transforms]、[timm Mixup][timm-mixup] | v2/v3 alpha=0.2，v4 alpha=0.8；v2/v3 随机选 Mixup 或 CutMix，v4 使用 timm batch 模式。 |
| CutMix | Yun 等，[CutMix][cutmix-paper]；同上两种 batch 实现 | alpha=1.0；切换概率、配对方式和边界处理跟随各自实现，不等同于在 Mixup 后继续执行 CutMix。 |
| RandomErasing | Zhong 等，[Random Erasing Data Augmentation][erase-paper]；[timm erasing 实现][timm-erase] | v2 p=0.1，CPU 在 Normalize 前用常量 0 擦除，DALI 用归一化后的黑色；v4 p=0.25、Normalize 后逐像素 Gaussian 填充。v3 不擦除。 |
| Repeated Augmentation（RA） | Hoffer 等，[Augment your batch][repeat-paper]；[DeiT sampler][deit-sampler] → [torchvision RASampler][tv-sampler] | v2 参数 4、v3 参数 3。PyTorch DDP 的重复后截断来自参考 sampler；**非 DDP 的完整重复和 DALI 多遍读取是 LLR2 行为**，不是论文要求每 epoch 读完整数据集 3/4 次。v4 RA1。 |
| Label smoothing | Szegedy 等，[Rethinking the Inception Architecture][smoothing-paper]；具体系数来自各自训练参考 | v2 0.1 对应 torchvision 改进配方；v3 0.11 对应 torchvision ViT-B/32；v4 0.1 对应 ConvNeXt，由 timm 混合标签实现，CE 不再重复平滑。 |

**后端来源不能只看相似名称：**v3 的 CPU 路径使用 torchvision `AutoAugmentPolicy.IMAGENET`；
DALI 路径历史上使用 NVIDIA `get_image_net_policy()`（`image_net`）。两者不是同一个子策略表：
例如 torchvision 第一项是 Posterize / Rotate，而当前 DALI 第一项是 Equalize / ShearY。
DALI 还提供 `get_reduced_image_net_policy()`，但**当前 LLR2 没有使用它**。
因此 v3 的后端差异除了插值、RNG 和 RA 预算，还包括 AutoAugment 策略内容；
历史结果应保留实际后端，不能将本次补充来源说明当作修改策略表的理由。
`equalize` 弃用修复只替换当前 DALI 策略内的算子接口，没有把策略表换成 torchvision 的版本。

**实现版本：**本次逐项核对的源码快照为 torchvision v0.28.0、timm v1.0.29、DALI v2.3.0，
ConvNeXt 仓库 `048efcea897d999aed302f2639b6270aedf8d4c8`。
[ConvNeXt 原始安装说明][convnext-install] 记录论文使用 timm 0.3.2；本项目调用当前安装的 timm，
所以 preset 4 表示按原配方配置的现代实现与 DALI 适配，不承诺复现原始软件栈。
依赖升级时需重新核对操作池和默认参数；引用的 tag 不会自动锁定运行环境版本。

## 2. v3：旧 ViT 的完整训练配方

在 `7484fdc` 中，ViT 仅接受 preset 0/1，preset 2 会报错；0 和 1 都进入同一套配置。
当前 `preset=3` 恢复这套完整配置，包括原 transforms API、batch collate、sampler 和损失。
默认图像处理流程（torchvision 后端）如下；DALI 的策略来源差异见第 1.2 节：

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
而 SURF 的 ViT 提交脚本现默认 `preset=3`，恢复历史的三倍 DALI 训练量。

PyTorch 和 DALI 使用各自的随机裁剪、插值、增强和 RNG 实现，也不保证逐位一致。
当前共享 DALI pipeline 是 CPU JPEG 解码，之后在 GPU 做缩放、增强和归一化；
随机擦除的矩形采样/边界处理也与 torchvision 实现不同。保持配方选项一致不代表像素完全相同。

DALI 2.3 自带的自动增强仍调用废弃别名 `fn.experimental.equalize`。共享 pipeline 通过公开的
自定义 policy/suite 接口，把 Equalize 替换为正式的 `fn.equalize`（旧 DALI 无此接口时保留兼容路径）；
不改动安装包或全局屏蔽警告。策略名称、操作顺序、概率和幅度保持原样，固定种子的实际 GPU
图像/标签已与修复前管线逐项对照一致；此接口修复不构成新的增强配方。

**普通训练的 DALI 生命周期：** 新运行保留训练和验证 pipeline，跨完整 epoch 复用 reader、
增强图、解码线程和预取缓冲区，随机流连续推进。每个逻辑 epoch 仍执行相应的 RA3/RA4 预算；
末尾只补读未结束的 reader 尾部以完成 reset，补读样本不进入训练。训练 checkpoint 保存各 rank 的
训练/验证 loader 状态（包括 DALI reader、增强 RNG 和 iterator），恢复后可继续同一随机流。
旧 checkpoint 缺少 loader 状态时，显式使用逐 epoch 重建/重新设种子的兼容模式，避免改变旧 DDP
实验的后续随机流；旧单卡 DALI 的未保存 RNG 无法精确重放。
`run_config.dali_loader.pipeline_lifecycle` 及日志记录 `persistent` / `epoch_seeded`。
新的训练 checkpoint 使用格式 `checkpoint_version=2`，仍能读取旧格式 1；此版本号与 ViT preset
语义版本独立，旧版训练程序会拒绝新版 checkpoint，避免忽略 reader 状态后静默改变随机流。
单卡显式 `num_samples` 子采样仍需重建；正常完整训练无需重建。
新生命周期不更改 v3 的增强选项、RA3、标签平滑或步数，但其随机序列与旧逐 epoch 设种子方案不同。
该变更针对普通训练，misalignment 自己的 pass 生命周期不由此推断。

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
默认值都已改成 v3。ConvNeXt-Tiny/21k-p 也显式支持 preset 3 作为历史增强对照；
其他模型不能据此推断支持 preset 3。

v3 恢复增强、采样和损失配置；历史比较还需对齐数据分区、后端、batch、总更新次数、
optimizer/scheduler、随机种子和精度。普通训练的新运行默认 BF16 评估；若历史指标来自 FP32，
可传 `--eval_precision fp32`，SURF 对应 `LLR2_EVAL_PRECISION=fp32`。
恢复相同配方不等于跨后端、跨 GPU 数或跨软件版本逐位复现。

这些 ImageNet level 的**图像操作来源**与第 1.2 节相同；level 编号、A/B 对称使用及禁止
21k-p level 2 是本项目的实验接口设计，不是 torchvision 或相关论文的通用增强等级。

**当前普通训练预算：** 四种 ViT 均采用 AdamW，ImageNet-1k 峰值 LR 为 `5e-4`，21k-p 为 `3e-4`，
weight decay `0.05`、总计 90 epochs（前 5 epochs 线性 warmup，后 85 epochs cosine）。
这是当前本机的训练预算；历史 ImageNet-1k 的 300 epochs / 30 warmup 和此前 21k-p 的 `5e-4` LR
应按实际运行源码识别，不能因增强同为 v3 而视为同一训练预算。恢复旧 checkpoint 仍加载保存的
optimizer/scheduler 状态，不会自动把旧调度压缩成 90/5。misalignment 的调度参数由自己的入口控制。

## 6. ConvNeXt-Tiny / 21k-p：preset 4 与 v3

ConvNeXt/21k-p 显式支持 0/1/2/3/4，0/1/2 复用 ViT 的 224 输入版配方，3 完整复用
历史 v3 图像增强、collate、RA3 和损失。**4 是新的 ConvNeXt 配方，3 的含义不变。**
现有 ConvNeXt/1k 入口保持原样。SURF ConvNeXt sbatch 默认 4。

| 项目 | preset 3（完整历史配方） | ConvNeXt preset 4 |
|---|---|---|
| 裁剪 / 翻转 | RRC224 / p=0.5 | RRC224 / p=0.5 |
| 初始缩放插值 | bilinear | bicubic |
| 单图增强 | ImageNet AutoAugment | timm `rand-m9-mstd0.5-inc1` |
| RandAugment | 无 | 抽取 2 个操作，各以 p=0.5 应用；Gaussian(9,0.5) 幅度截断到 [0,10] |
| Mixup / CutMix alpha | 0.2 / 1.0 | 0.8 / 1.0 |
| Batch 配对 | torchvision v2 RandomChoice / 循环移位 | timm batch Mixup / 反转配对 |
| 随机擦除 | 无 | p=0.25，归一化后逐像素 Gaussian(0,1) 填充 |
| DALI 数据量 | RA3，约 3N/epoch | RA1，约 N/epoch |
| 训练平滑 | CE 内 label_smoothing=0.11 | timm 混合标签内平滑 0.1，CE 自身为 0，避免重复平滑 |
| 验证损失 | CE 含 smoothing=0.11 | 普通 CE，无 smoothing |
| 验证缩放 | bilinear | bicubic |

差距较大，尤其 RA3 改变三倍数据量。需要历史增强对照时显式 `LLR2_PRESET=3`，
并同时对齐总样本量、microbatch、有效 batch 和 optimizer 更新次数。两者 loss 定义不同，
不能直接比较绝对 loss。参考 [官方配方表 5](https://arxiv.org/html/2201.03545v2#A1.SS1)、
[官方增强默认值](https://github.com/facebookresearch/ConvNeXt/blob/main/main.py)。

CPU preset 4 直接使用 timm `create_transform()` 和 `Mixup()`。DALI 复用相同 15 种
increasing 操作池、操作概率、Gaussian 幅度、随机符号和像素擦除，并使用正式 equalize/filter。
它与 PIL 不逐像素相同：幅度量化到 0.01，几何 warp/rotate 受 DALI 限制采用 bilinear、
标量边界填充 128；初始 resize 和验证 resize 仍为 bicubic。PIL 的几何操作使用 bicubic 和
ImageNet RGB 均值填充，颜色/滤波/随机裁剪也由各后端实现。这是明确的 GPU 适配。

官方参数 `color_jitter=0.4` 也传给 timm，但本次核对的 timm 在开启该 RandAugment、
且未设置 `force_color_jitter` 时会禁用独立 ColorJitter。因此实际 pipeline 没有在 RandAugment
之外再叠加一次 ColorJitter；颜色变化仍可由其操作池采样得到。不要只根据传入参数罗列实际操作。

ConvNeXt/21k-p 随机初始化（weights=None），初始化回调保留截断正态 std=0.02、
LayerScale=1e-6 和 LayerNorm=1/0，不使用 Linear/Conv 的通用 reset 分布。
Tiny stochastic depth=0，不用 EMA 或梯度裁剪。AdamW 的矩阵权重 wd=0.05，
bias/LayerNorm/LayerScale 不衰减（torchvision 的 LayerScale 形状 [C,1,1] 也排除）。
90 epochs / 5 warmup / 85 cosine，默认 microbatch=512、累积 8 次、有效 batch=4096，
peak LR=0.004、min LR=1e-6；改有效 batch 时两个 LR 端点按 batch/4096 缩放。
调度按实际 optimizer 更新次数计算，末尾不足一个累积组也正确归一化。
checkpoint 保存累积次数、recipe version、optimizer/scheduler、DALI 和各 rank RNG 状态。
本配方来源是完整 ImageNet-22k，使用 winter21 21k-p 是数据集适配，不宣称严格复现论文准确率。
运行说明见 [ConvNeXt SURF 配置](../../tool/snellius/README_convnext.md)。

## 7. 后续 LLM 如何识别和描述旧实验

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

## 8. 代码与验证入口

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
- [test_distributed_dali.py](../../test/test_distributed_dali.py)：实际 DALI 1/2/4 分片跨轮复用、RA3 尾部、
  reader/RNG 状态恢复、旧 checkpoint 兼容和普通训练连续运行/断点恢复后的权重对照。

后续若修改配方、默认尺寸、采样预算或 checkpoint 语义，应同步更新本文和对应测试。
新增或修改增强时，同时记录方法论文、实际参考代码的版本 / 链接、LLR2 改动和后端差异；
无法从历史记录确认的来源应明确写“未记录 / 未确认”，不能根据相似参数补造出处。

<!-- Pinned reference snapshots; these links document provenance, not runtime dependency pins. -->
[tv-presets]: https://github.com/pytorch/vision/blob/v0.28.0/references/classification/presets.py
[tv-vit]: https://github.com/pytorch/vision/blob/v0.28.0/references/classification/README.md#vit_b_32
[tv-recipe]: https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
[tv-autoaugment]: https://github.com/pytorch/vision/blob/v0.28.0/torchvision/transforms/autoaugment.py
[tv-transforms]: https://github.com/pytorch/vision/blob/v0.28.0/references/classification/transforms.py
[tv-sampler]: https://github.com/pytorch/vision/blob/v0.28.0/references/classification/sampler.py
[llr2-import]: https://github.com/twoentartian/LLR2/blob/98adddec9d8e60edb91db81dab1e53a302d9802d/py_src/ml_setup_dataset/dataset_imagenet.py
[llr2-vit-origin]: https://github.com/twoentartian/LLR2/commit/c4de35f6ef7667e6b208c4a15697ee7492fb35df
[llr2-legacy-vit]: https://github.com/twoentartian/LLR2/blob/7484fdcad60dd8c2c67d33f2a9fb5d139180f0a3/py_src/ml_setup/vit.py
[convnext-paper]: https://arxiv.org/html/2201.03545v2#A1.SS1
[convnext-main]: https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/main.py
[convnext-datasets]: https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/datasets.py
[convnext-install]: https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/INSTALL.md
[timm-transforms]: https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/transforms_factory.py
[timm-aa]: https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/auto_augment.py
[timm-mixup]: https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/mixup.py
[timm-erase]: https://github.com/huggingface/pytorch-image-models/blob/v1.0.29/timm/data/random_erasing.py
[dali-aa]: https://github.com/NVIDIA/DALI/blob/v2.3.0/dali/python/nvidia/dali/auto_aug/auto_augment.py
[dali-ra]: https://github.com/NVIDIA/DALI/blob/v2.3.0/dali/python/nvidia/dali/auto_aug/rand_augment.py
[dali-ta]: https://github.com/NVIDIA/DALI/blob/v2.3.0/dali/python/nvidia/dali/auto_aug/trivial_augment.py
[deit-sampler]: https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/samplers.py
[aa-paper]: https://arxiv.org/abs/1805.09501
[ta-paper]: https://arxiv.org/abs/2103.10158
[ra-paper]: https://arxiv.org/abs/1909.13719
[mixup-paper]: https://arxiv.org/abs/1710.09412
[cutmix-paper]: https://arxiv.org/abs/1905.04899
[erase-paper]: https://arxiv.org/abs/1708.04896
[repeat-paper]: https://arxiv.org/abs/1901.09335
[smoothing-paper]: https://arxiv.org/abs/1512.00567
