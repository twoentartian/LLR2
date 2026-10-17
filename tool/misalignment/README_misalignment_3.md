# misalignment_3：partition 梯度夹角实验

入口为 `tool/misalignment_3.py`，从 `misalignment_measurement_2.py` 独立复制。
保留原有数据等分、双方向训练、优化器、AMP/compile、checkpoint、方差和可选 flatness 流程。
原入口和公共梯度组合代码没有修改。

## 训练时测量

可沿用 misalignment_2 的参数，例如从仓库根目录运行：

```bash
python tool/misalignment_3.py \
  --dataset cifar10 --model_type resnet18_bn \
  --epochs 100 --batch_size 128 --random_seed 42 \
  --angle_batch_size 256 --angle_interval 1 \
  --stability_angle_tolerance 5 --stability_patience 3 \
  --skip_relative_flatness -o runs/misalignment_3_train
```

每个方向从相同初始化开始，更新前测一次，再在每个指定 epoch 更新结束后测量。
初始化记为 `epoch=-1`，训练 epoch 从 0 开始；最后一个 epoch 一定测量。
即使 `--eval_last_10_percent` 跳过常规 loss/accuracy 评估，夹角仍按自己的频率测量。

在同一模型参数位置，分别用 microbatch 累积两个 partition 的平均 loss 梯度：

```text
g_A = ∇θ mean(loss on partition A)
g_B = ∇θ mean(loss on partition B)
raw_gradient_angle_deg = acos(dot(g_A, g_B) / (||g_A|| ||g_B||)) * 180 / pi
```

先按样本数累积梯度，再计算夹角；不是平均 minibatch 夹角。
默认使用完整 partition、eval 模式和 FP32，不使用训练 AMP 的缩放梯度。
数据增强固定在 epoch 0，测量不会修改训练的增强 epoch、模型 buffers、`.grad` 或随机数状态。
多卡使用不补齐、不丢弃的评估分片，先求和梯度，再计算全局夹角。

`angle_stable` 默认要求**原始 A/B 梯度夹角至少 175°，连续出现 3 次训练测量**。
初始点和随机参数点不计入连续次数；无效夹角会清零连续次数。该标记不会自动停止训练。
零梯度或非有限梯度的夹角记为 NaN，不将它们当成 180°。

这是待验证的角度判据。原优化器做 train 下降 / val 上升：原始梯度接近 180° 时，
`g_train - g_val` 不会抵消，因此该标记不能直接证明原优化器已经收敛。

## 只测量参数空间随机点

```bash
python tool/misalignment_3.py \
  --dataset cifar10 --model_type resnet18_bn --random_seed 42 \
  --probe_only --random_parameter_points 100 \
  --random_parameter_radius 0.1 --angle_seed 31415 \
  --angle_batch_size 256 -o runs/misalignment_3_random
```

此模式不构建 optimizer、不训练、不测 flatness。
先测初始化位置，再独立采样 N 个参数位置；每个位置都测量两个 partition。
仅对可训练参数添加扰动，分布明确为：

```text
theta_i = theta_initial + radius * ||theta_initial|| * z_i / ||z_i||
z_i ~ N(0, I)
seed_i = angle_seed + 10000 + i, i = 1, ..., N
```

即以初始化为中心、指定相对 L2 半径的球面采样，每个点的 buffers 都恢复为初始值。
`--random_parameter_radius 0.1` 表示扰动范数为初始可训练参数范数的 10%。
可用不同半径分别运行，比较局部和较远位置的夹角分布。
去掉 `--probe_only` 即可先完成同样的随机点测量，再从原始初始化开始正常训练。

`--angle_samples -1`（默认）使用完整 partition。
若完整数据测量成本过高，可设置 `--angle_samples 10000`；这是每侧固定的随机数据子集，
与参数空间中的随机点数量 `--random_parameter_points` 独立。抽样不放回，超过 partition 大小时使用全部数据。

## 输出

- `gradient_angles_<direction>.csv`：每个训练方向的初始化和 epoch 结束测量。
- `gradient_angles_initial.csv`：启用随机参数点或 `--probe_only` 时的初始点及随机参数点测量。
- `gradient_probe_indices_<label>.pt`：每侧在原 A/B partition 内使用的数据索引。
- `summary.json`：测量配置、球面采样定义、每次全局测量及原训练结果。
- `initial.model.pt`：只测量模式的采样中心；训练模式保留原有每方向初始/最终 checkpoint。

CSV 中 `parameter=__all__` 表示把所有可训练参数的原始梯度视为一个向量计算的**全局夹角**，
其余行是逐参数张量的夹角。稳定判据使用全局夹角。
`raw_gradient_cosine` / `raw_gradient_angle_deg` 是主要指标，
`descent_ascent_angle_deg` 是额外保存的 `-g_A` 与 `+g_B` 的夹角，等于 `180 - raw_gradient_angle_deg`。
每个点同时记录两侧梯度范数、loss、accuracy、loss/accuracy gap、采样半径和测量时间，
方便检验角度与 misalignment 的关系；脚本不预设夹角到 misalignment level 的映射。

原有的 `gradient_geometry_*.csv` 继续记录训练 minibatch 的几何统计，含义与新的固定位置测量不同。
训练默认仍会做原来的相对 flatness 测量，可用 `--skip_relative_flatness` 延后。

回归测试：`python -m pytest -q test/test_misalignment_3.py`。
