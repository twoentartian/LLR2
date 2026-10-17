# 单节点多 GPU 训练

两个入口均可通过 `torchrun` 使用同一节点的 2 或 4 张 CUDA GPU。
直接运行 `python tool/<入口>.py ...` 仍使用原有单设备模式。
从仓库根目录运行，并先激活安装了匹配 CUDA 的 PyTorch、NCCL 和 DALI 的环境。

## 高准确率模型

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tool/generate_high_accuracy_model.py \
  -m vit_b_32 -d imagenet1k -n 1 -w 1 \
  --batch_size 512 --dali --amp --enable_eval -o vit_ddp
```

多卡模式仅支持单个模型 (`-n 1 -w 1`)，当前支持通过 StandardAdapter
训练的分类模型。`-w` 仍表示独立模型并发数，不表示 GPU 数。
Lightning、Diffusion 和自定义训练 adapter 在多卡模式下会明确报错。

`--batch_size` 是所有 GPU 合计的 mini-batch 大小，必须能被 GPU 数整除；
上例四卡每卡 128。省略时使用模型配置的 batch size，含义同样为全局大小。
有梯度累积时，有效更新 batch 再乘以累积次数。学习率不会自动缩放。
`--core` 是整个作业的 CPU 核心预算，进程间分配后每进程上限仍为 8。
已有自定义 sampler（例如 ViT 的 repeated augmentation sampler）保留其采样规则。

每个进程使用一个 DDP 副本。训练结果按样本数汇总，验证集精确分片、无重复计数。
日志 CSV 和模型只由 rank 0 保存，保存的权重不含 DDP/compile 包装前缀。
训练 checkpoint 另保存各 rank 的随机状态，恢复时必须使用相同进程数：

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tool/generate_high_accuracy_model.py \
  --load_checkpoint vit_ddp/0/training_checkpoint.pt
```

恢复沿用 checkpoint 的配置，不再重复传 `--dali`、`--amp` 或 `--batch_size`。
DALI 的分布式训练 pipeline 按 epoch 重新设定 reader/augmentation seed，
可以直接恢复到对应 epoch。PyTorch 多 worker 的随机增强仍不保证逐位重放。
分布式 DALI 暂不支持 `DataloaderConfig.num_samples` 子采样。

## Misalignment

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tool/misalignment_measurement_2.py \
  --dataset imagenet1k -m vit_b_32 --batch_size 512 \
  --loader_backend dali --random_seed 42 \
  --skip_relative_flatness -o misalignment_ddp
```

`--batch_size` 是每个分区的全局 batch：每一步分别处理全局 512 个 A 样本和
512 个 B 样本。`--eval_batch_size` 同样是全局大小。
所有 rank 共享初始化、split seed 和 A/B 划分；dropout 使用各 rank 的随机流，
两个方向会重新使用相应的随机流。

normalized_two_sided 先分别汇总原始 A/B 梯度，再执行已有的逐参数归一化、
加权和重缩放。梯度通过分桶 all-reduce 汇总，不依赖 DDP 的 backward hook。
difference / mean_relative_gap 同样支持分布式梯度更新。
loss、accuracy、tokenizer 非零计数、ViT token 范数及直方图均按全局数据汇总。
BatchNorm 使用每卡的 batch 统计量，buffer 从 rank 0 同步；没有自动转换为
SyncBatchNorm，因此不保证与单卡前向等价。浮点归约顺序、dropout、DALI 随机增强
也会导致单卡/多卡运行不逐位相同；归一化方向近乎抵消时对舍入尤其敏感。

为保证各 rank 训练步数相同，训练 sampler 在数据量不能被 GPU 数整除时最多补齐
`GPU 数 - 1` 个样本；验证集不补齐。misalignment 的 summary 记录补齐数量。
DALI 的 A/B manifest 在解码之前按同样的固定序列分片，保持配对关系，
每张 GPU 使用自己的 CPU 解码/GPU 预处理 pipeline。

多卡模式默认跳过 relative flatness；单设备默认保留原有测量行为。
`--skip_relative_flatness` 或 `--no-relative_flatness` 可以显式跳过。
跳过时仍保存初始模型、最终模型、gap checkpoint、划分、种子和训练指标；
JSON 将 flatness 标记为 `deferred`，相关测量值为 `null`。
之后可以在单 GPU 上使用已有的 `--flatness_from_checkpoint` 路径测量最终模型，
并复用该 run 的 `split_seed`、`random_seed`、augmentation 等设置。
本轮没有增加多卡二阶测量。

ViT 的 baseline、register、RoPE、SwiGLU 四种变体都支持 ImageNet-21k-p。
使用数据集名 `imagenet21k_p`，也接受 `imagenet21k-p`、`imagenet21kp`、
`imagenet21k`；这些名字均指向项目已有的 21k-p 数据布局，分类头与 soft-label
宽度根据训练目录中的实际类别数构建（现有服务器数据为 10,450 类）。
默认根目录为 `~/dataset/imagenet21k_resized`，其下为 `imagenet21k_train/`
与 `imagenet21k_val/`；可通过 `py_src/ml_setup_dataset/dataset_env.py` 中的
`imagenet21k_p_path` 覆盖。两个 split 必须具有相同的类别目录及标签映射。

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tool/generate_high_accuracy_model.py \
  -m vit_b_32 -d imagenet21k_p --batch_size 512 --dali --amp --enable_eval

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tool/misalignment_measurement_2.py \
  --dataset imagenet21k_p -m vit_b_32 --batch_size 512 \
  --loader_backend dali --augmentation 1 --skip_relative_flatness
```

21k-p 普通 ViT 训练复用现有 ViT recipe（224 输入、ImageNet AutoAugment、
Mixup/CutMix、label smoothing 0.11、AdamW、lr 5e-4、weight decay 0.05），
训练预算采用项目已有的 21k-p 设置：90 epochs、5 epochs warmup 后 cosine。
`-e` 可覆盖普通训练的 epoch 数，已有 scheduler 的总预算仍来自 recipe。
misalignment 不使用普通训练的 batch Mixup/CutMix，21k-p 仅支持 augmentation
`none/0` 与 `1`；`2` 会在加载数据前报错。其默认 optimizer 为 AdamW，
lr 5e-4、weight decay 0.05、90 epochs、5 epochs warmup，scheduler 默认仍为
fixed，可通过现有 CLI 参数覆盖。ImageNet-1k 的原有设置保持不变。

## Snellius

仓库提供 `tool/run_single_node_multi_gpu.sbatch`，默认申请一个 H100 节点的四卡。
只启动一个 Slurm task，再由 torchrun 启动四个进程，避免双重启动：

```bash
sbatch tool/run_single_node_multi_gpu.sbatch \
  tool/generate_high_accuracy_model.py \
  -m vit_b_32 -d imagenet1k --batch_size 512 --dali --amp --enable_eval

sbatch --gpus-per-node=2 --cpus-per-task=32 \
  tool/run_single_node_multi_gpu.sbatch \
  tool/misalignment_measurement_2.py \
  --dataset imagenet1k -m vit_b_32 --batch_size 512 \
  --loader_backend dali --skip_relative_flatness
```

按需要调整 partition、walltime、account 和数据路径。A100 可覆盖
`--partition=gpu_a100`。不要手动覆盖 Slurm 设置的 `CUDA_VISIBLE_DEVICES`。

## 验证

```bash
python -m pytest test/test_distributed_training.py -q
python -m pytest test/test_distributed_dali.py -q
```

第一组在 CPU/Gloo 上实际启动 2/4 个进程，比较单进程与分布式的参数更新、指标、
checkpoint 和两个 misalignment 方向。第二组需要一张 CUDA GPU 与 DALI，
依次运行 2/4 个真实 reader 分片，检查训练尾批次和评估样本覆盖。
这不能替代 Snellius 上实际的多 GPU NCCL、吞吐量和 torch.compile 验证。
