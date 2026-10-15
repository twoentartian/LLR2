# LLR2 重点文件说明

本文档面向新的 session 和 LLM 工具，说明 LLR2 中最常用的实验入口、结果处理脚本和模型配置依赖。项目主要使用 PyTorch，研究内容集中在高准确率模型、grokking、misalignment 以及模型权重/损失地形分析。

## 使用约定

- 通常应从仓库根目录运行脚本。许多 `tool/*.py` 文件会把仓库根目录加入 `sys.path`，因此直接从任意目录运行可能导致数据路径或相对输出路径不同。
- 模型 checkpoint 通常由 `py_src/model_opti_save_load.py` 保存，外层是包含 `state_dict`、`model_name` 和 `dataset_name` 的字典，而不一定是裸 `state_dict`。
- `py_src.ml_setup.factory.get_ml_setup_from_config()` 是多数训练和分析脚本进入模型/数据配置的入口；它返回 `MLSetup`，其中包含模型、adapter、数据集、criterion 和 dataloader 默认配置。
- 大多数实验输出目录会保存 `command.txt`，并用 CSV 记录训练或评估轨迹。分析结果时应先读取该文件和同目录的 JSON 元数据。

## 主干文件

### `tool/generate_high_accuracy_model.py`

高准确率分类/生成模型 checkpoint 的通用训练入口。

- 根据模型和数据集配置建立 `MLSetup`，然后训练一个或多个模型；支持 CPU/CUDA、AMP、`torch.compile`、DALI、验证集评估和多进程并行。
- 支持从头初始化、指定初始模型、迁移学习、从 `training_checkpoint.pt` 恢复，以及用另一个模型构造 opposite-direction 初始化。
- 每个模型会写出 `.model.pt`、`.optimizer.pt`、`*.log.csv`；启用中间保存时还会写 epoch checkpoint。输出目录还包含 `info.json`、训练配置和可恢复训练状态。
- `training_model()` 是单个模型的主要训练流程；`save_training_checkpoint()` / `load_training_checkpoint()` 负责可恢复训练。
- `torchrun --standalone --nproc-per-node=2/4` 支持单节点、单模型 DDP 分类训练和 DALI；batch size 为全局大小，rank 0 写文件，checkpoint 保存各 rank RNG。公共运行时在 `py_src/distributed.py`，用法与 Snellius 脚本见 `tool/README_multi_gpu.md`。

### `tool/generate_grokking.py`

模运算/算术任务的 grokking 训练入口，主要使用 `transformer_for_grokking`。

- 数据可以从已有的 `train.txt`/`val.txt`/`tokenizer.txt` 读取，也可以通过 `--dataset_exp` 生成；支持随机、结构化 split、噪声和 chessboard transpose 等设置。
- `GrokkingParameters` 保存训练配置；`train_grokking()` 执行训练、验证、AdamW + warmup/cosine 学习率调度、checkpoint 保存和可选的早停策略。
- 支持 inverse train/val 模式：用同一初始模型分别训练两个分区，便于比较方向性差异。
- 常见输出包括 `command.txt`、`*.log.csv`、模型 checkpoint、`final_correct_position.csv`，以及可选的权重统计和 distance-to-origin 服务记录。
- `generate_grokking_phase_diagram.py` 复用本文件中的数据生成、模型构建和训练逻辑。

### `tool/generate_grokking_phase_diagram.py`

执行 grokking 学习率（log-spaced）× weight decay（linear-spaced）网格扫描。

- 每个 `(learning_rate, weight_decay)` 网格单元调用 `train_cell()`，实际训练由 `generate_grokking.train_grokking()` 完成。
- `read_phase_from_log()` 根据训练/验证准确率把单元分类为 `confusion`、`memorization`、`grokking` 或 `comprehension`。
- 已完成的单元可以通过已有 `00.log.csv` 跳过；连续 confusion 单元可以触发后续 weight decay 跳过。
- 输出根目录包含 `command.txt`、`grid_spec.json`；每个网格单元通常有 `meta.json` 和 `00.log.csv`。

### `tool/generate_grokking_phase_diagram_subdataset.py`

为多个规模的嵌套 modular 子数据集准备 phase-diagram 批量实验的编排工具。

- 它本身主要负责数据准备和写 runner，不直接执行模型训练。
- 从完整 master 数据集复制出多个前缀：保留 `x < n` 且 `y < n` 的样本，并复制 tokenizer；每个子数据集写入 `prefix_metadata.json`。
- 根据训练样本数和 batch 数计算每个规模的 batch size、epoch 数和结果目录，最后写出 `manifest.json`、`run_all.py` 和 `run_all.sh`。
- 生成的 runner 会逐个调用 `generate_grokking_phase_diagram.py`。修改 phase-diagram 参数时，应同时检查这里传递给 runner 的参数。

### `tool/misalignment_measurement_2.py`

对两个数据分区进行对称 train/validation misalignment 测量的主程序。

- 对图像数据会先合并训练集和测试/验证集，再用 `split_seed` 打乱并等分为 A/B；A/B 不等同于原始数据集的 train/val。
- 默认执行两个方向：A 作 train、B 作 val，以及 B 作 train、A 作 val；每个方向从同一个初始模型状态开始。
- 支持 `normalized_two_sided`、raw difference 和 mean-relative-gap 目标；normalized two-sided 路径会对 train/val 梯度逐参数归一化、加权、合并并重缩放。
- 训练过程中写出每 epoch 的优化指标、梯度几何和权重方差；结束后对指定层执行 Hutchinson/Hessian 相对平坦度测量。
- 每 epoch 的日志和优化 CSV 还包含 `running_train_loss` / `running_val_loss`，复用训练 forward 的原始 loss 按样本数加权，跳过末尾 eval 时仍汇报；val 权重为 0、未执行该侧 forward 时记 `NaN`。完整 eval 的日志字段为 `eval_train_loss` / `eval_val_loss`，CSV 仍为 `loss_a` / `loss_b`。
- 会保留 accuracy gap 和 loss gap 最大的 checkpoint，并对这些 checkpoint 追加相对平坦度测量。
- 单节点 2/4 卡通过显式 all-reduce 同步原始 train/val 梯度后再归一化组合，同时支持 PyTorch/DALI 分片。多卡默认延后 flatness，单设备可用 `--skip_relative_flatness` 显式延后；仍保存最终模型、gap checkpoint、split seed 等，summary 记 `deferred`/`null`。多卡 flatness 尚未实现。
- `--eval_last_10_percent` 可将 epoch 末全量 A/B 评估限制到最后 `ceil(epochs/10)` 个 epoch（默认每 epoch 评估）。跳过时优化 CSV 标记 `evaluated=0`，loss/accuracy/gap 为 `NaN`；范数和 gap checkpoint 仅记录已评估 epoch，权重方差和梯度几何照常记录。对称目标更新所需的 baseline eval 保留。
- 主要输出为 `optimization_*.csv`、`gradient_geometry_*.csv`、`weight_variance_*.csv`、`relative_flatness_*.csv`、`selected_relative_flatness_*.csv`、模型 checkpoint 和 `summary.json`。
- `summary.json` 还记录数据分区、loader、AMP/compile、optimizer、checkpoint 选择和 `git_provenance`。最近一次修改后，Git commit、branch 和 tracked working-tree 状态会在运行时写入结果。
- CCT 专用运行通常通过同目录的 `tool/misalignment_cct_entry.py` 启动；该 wrapper 会覆盖 CCT 的输入尺寸和优化器 recipe，再调用本文件。

## 工具性文件

### `tool/linear_interpolate_models.py`

评估两个 checkpoint 之间的参数线性插值路径，不进行训练。

- 读取两个 `.model.pt` 或兼容 checkpoint，检查 state-dict key、shape 和 dtype。
- 对浮点 tensor 使用 `torch.lerp` 生成 `alpha=0..1` 的插值模型，对每个插值点计算 train/validation loss 和 accuracy。
- 输出 `interpolation_metrics.csv`；适合检查两个模型之间是否存在低损失连接路径。

### `tool/calculate_relative_loss_landscape_sharpness.py`

计算模型最后一个 `nn.Linear` 分类头的相对 loss-landscape sharpness。

- 输入可以是单个模型 checkpoint，也可以是保存多个 tick 的 LMDB 目录。
- 通过 factory 建立模型和数据配置，先用 engine 做 baseline validation，再用最终线性层的解析 Hessian/Gram 公式计算 `kappa`。
- 当前实现要求交叉熵 `reduction="mean"`，并要求模型存在会参与 forward 的最后一个 `nn.Linear`。
- 结果打印到终端，并追加写入 `<checkpoint>.relative_loss_landscape_sharpness.log`。

### `result_processing_tool/calculate_cosine_similarity.py`

比较两个 PyTorch checkpoint 的参数余弦相似度和范数。

- 能读取裸 `state_dict` 或包含 `state_dict`、`model_state_dict`、`model` 等常见字段的 checkpoint。
- 支持去除 `module.` 前缀、严格 key/shape 检查、正则筛选 key、跳过非浮点 tensor。
- 输出所有公共 tensor 的整体 flatten cosine、两侧范数，以及按 layer 分组的 cosine、范数比例和估计参数量。
- 主要面向命令行诊断，不写结果文件；LLR2 自己保存的 checkpoint wrapper 可以直接读取。

## 重要依赖库文件

### `py_src/ml_setup/factory.py`

模型/数据集配置的总路由。

- `get_ml_setup_from_config(model_type, dataset_type, preset, ...)` 把字符串转换为 `ModelType`/`DatasetType`，并调用具体的 setup builder。
- `_build()` 根据模型和数据集组合选择 LeNet、ResNet、CCT、Transformer/grokking、扩散模型等配置。
- 这里还集中处理 ImageNet DALI setup 的选择和 `dali_device_id` 传递。
- 高准确率训练和 misalignment 的 ImageNet DALI 图像操作共同使用 `py_src/ml_setup_dataset/dataset_imagenet.py` 的 `build_dali_imagenet_pipeline()`：CPU 解码、GPU 缩放和归一化。misalignment loader 只适配 A/B 文件列表、固定顺序和每轮 seed。
- 当一个模型和数据集组合没有实现时，会抛出 `NotImplementedError`；分析脚本遇到 “no setup” 时应先检查这里的分派逻辑。
- CCT14 支持 `cct_14_7x2_224` + `imagenet21k_p`（CLI 也接受 `imagenet21k-p`）；入口为 `py_src/ml_setup/cct.py` 中的 `cct14_7x2_imagenet21k_p()`。默认数据根目录为 `~/dataset/imagenet21k_resized`，读取 `imagenet21k_train/` 和 `imagenet21k_val/`，分类头按实际类别目录数构建（服务器上为 10,450 类）。根路径可通过 `dataset_env.py` 的 `imagenet21k_p_path` 覆盖。

### `py_src/complete_ml_setup.py`

集中定义标准训练预设。

- `FastTrainingSetup.get_optimizer_lr_scheduler_epoch()` 根据 model、dataset 和 preset 返回 optimizer、scheduler 和 epoch 数。
- ViT-B/32 的 register 对照模型名为 `vit_b_32_reg4`，增加 4 个无位置编码的可学习 tokens；同 seed 下共同权重与 `vit_b_32` 初始化一致，通用训练 recipe 复用原版设置。
- 覆盖 SGD、Adam/AdamW、OneCycle、Cosine、MultiStep、warmup 等训练 recipe，并根据 dataloader 的 steps-per-epoch 换算 scheduler 步数。
- `TransferTrainingSetup.get_optimizer_lr_scheduler_epoch()` 定义允许的迁移学习组合和对应的优化器配置。
- 训练脚本的 `--preset` 默认值会影响学习率、weight decay、scheduler 和总 epoch；显式 CLI 参数通常会覆盖这些默认值。
- CCT misalignment launcher 会对 CCT 的 recipe 做额外覆盖，因此分析 CCT 实验时应同时查看 `tool/misalignment_cct_entry.py`。

## 建议的阅读顺序

ViT register 实验与 LayerNorm 前 token 范数 CSV 的说明见 `tool/misalignment/README_vit_registers.md`。misalignment 脚本默认对两种 ViT 启用范数记录，阈值由 `--token_norm_threshold` 控制，可通过 `--no-token_norms` 关闭。

1. 先看目标实验脚本的 `parse_args()` 和 `main()`，确认输入、输出和默认参数。
2. 再看 `py_src/ml_setup/factory.py`，确认字符串模型/数据集最终映射到哪个 setup。
3. 接着看 `py_src/complete_ml_setup.py`，确认 optimizer、scheduler、batch step 和 epoch 默认值。
4. 最后按输出目录中的 `command.txt`、JSON、CSV 和 checkpoint 文件核对实际运行参数；不要只根据文件夹名字推断实验配置。
