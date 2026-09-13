# 模型权重比较与排列对齐

在项目根目录运行以下命令。使用生成模型时的 Python 环境，依赖为
`torch`、`numpy`；PCA 另需 `matplotlib`，排列对齐另需 `scipy` 和项目的模型依赖。
只读取 checkpoint，不下载或加载训练数据集。文件格式复用
`py_src.model_opti_save_load`，默认匹配 `*.model.pt`，不读取 optimizer 文件。

## 所有模型对的逐层 cosine similarity

```bash
python3 result_processing_tool/calculate_pairwise_layer_cosine.py \
  tool/high_accuracy_BNN
```

默认输出 `tool/high_accuracy_BNN/layer_cosine_summary.csv`，一层一行。
N 个模型计算 N(N−1)/2 个不重复模型对：20 个模型为 190 对。
`mean_cosine` 为这些 pair-wise cosine 的算术平均值，不是平均权重的 cosine。
同时记录模型数、参数数、总 pair 数、有效/未定义 pair 数、总体标准差、最小值、最大值。
任一模型该层范数为零时，该 pair 的 cosine 为 NaN，排除出平均值并计数；
没有有效 pair 时统计值为 NaN。

```bash
python3 result_processing_tool/calculate_pairwise_layer_cosine.py \
  tool/high_accuracy_BNN -o result_processing_tool/bnn_layer_cosine.csv \
  --pairs-file result_processing_tool/bnn_all_pairs.csv
```

`--pairs-file` 可选，将每个 pair、每一层的结果另外记录下来。
默认将同一模块的 weight 和 bias 拼接成一个向量，排除 BatchNorm 的运行统计量。
这与 `plot_layer_weight_pca.py` 的分组规则一致。

常用选项：

- `--key-regex '^(conv|fc)'`：仅分析 BNN 的卷积/全连接层。
- `--exclude-bias`：仅排除 bias。
- `--exclude-regex REGEX`：排除匹配的参数。
- `--layer-level 2`：按模块名称的前两段合并分组。
- `--pattern '**/*.model.pt'`：递归收集模型。
- `--cache-dir PATH`、`--block-size 16384`：指定临时磁盘缓存位置、降低计算块大小。

数据先逐 checkpoint 写入 float64 磁盘缓存，缓存约占 `8 × 模型数 × 参数数` 字节。
计算每层的模型间 Gram 矩阵，不会同时把所有模型放进 Python 内存；
Gram 矩阵本身的内存随模型数平方增长。退出时删除缓存。
自定义浮点 buffer 无法仅从 state_dict 与参数完全区分，需要用正则筛选。
一个批次需要模型/数据集元信息一致、所选参数名称及形状一致。
输出文件已存在时拒绝覆盖，换一个 `-o` / `--pairs-file` 路径即可。

## 将 B 排列对齐到 A，保存为 C

单个或多个 B：

```bash
python3 result_processing_tool/permute_models.py \
  -a tool/high_accuracy_BNN/00.model.pt \
  -b tool/high_accuracy_BNN/01.model.pt tool/high_accuracy_BNN/02.model.pt \
  -o result_processing_tool/bnn_aligned
```

整个目录的 B：

```bash
python3 result_processing_tool/permute_models.py \
  -a tool/high_accuracy_BNN/00.model.pt \
  -b tool/high_accuracy_BNN \
  -o result_processing_tool/bnn_aligned
```

目录输入自动排除 A 本身。多个 B 逐一处理，分别保存，例如
`01.permuted.model.pt`、`02.permuted.model.pt`。目录递归输入保留相对目录结构。
原始 A、B 不修改；C 可由项目原有的模型加载函数直接读取。
每个 C 旁边有 `.model.json` 报告，包含来源、排列索引、迭代次数、是否收敛、
对齐前后全模型参数 cosine，以及内置模型的 B/C 前向输出检查结果。

内置支持 `bnn`、`bnn_floating`、`lenet4`、`lenet5`、`lenet5_large_fc`，自动读取 checkpoint 的
`model_name`。只有元信息缺失时才需 `--model-type`。
构建模型时复用 `ml_setup` 使用的模型类，不构建完整数据集 setup。

算法使用 [Git Re-Basin 的 weight matching 思路](https://arxiv.org/abs/2209.04836)：
对每个隐藏通道组做 Hungarian assignment，交替更新，最大化全模型参数的点积。
因为通道排列保持参数范数，该目标等价于最大化全模型参数 cosine、最小化 L2 距离。
迭代可能停在局部最优；不保证每一层单独的 cosine 都提高，也不改变 B 的模型能力。
`--max-iter 50` 控制最多扫描轮数，`--seed 0` 控制更新顺序，`--threads 4` 控制计算线程。
达到轮数上限仍会保存当前最好的排列，并在报告中设置 `converged: false`。

同步排列上一层的输出通道、下一层的输入通道、bias 和 BN 的可学习参数/运行统计量。
BN 运行统计量不参与匹配目标，但应用排列时随通道一起变化。
输入 RGB/像素顺序、输出类别顺序保持固定。卷积 flatten 到全连接时按整块通道排列，
例如 BNN 的 `fc1` 输入每个通道对应 9 个连续特征。
分析和排列的都是 BNN checkpoint 保存的浮点隐权重，不做额外二值化。
内置模型保存前会用 4 个固定随机输入在 float64 eval 下比较 B/C 输出；这不是准确率评估。

## 扩展其他模型

其他模型用 `--spec permutation.json` 描述合法的通道排列关系。例如两层 MLP：

```json
{
  "axes": {
    "fc1.weight": ["hidden", null],
    "fc1.bias": ["hidden"],
    "fc2.weight": [null, "hidden"],
    "fc2.bias": [null]
  }
}
```

每个列表与张量各维度一一对应；同名 group 必须使用同一排列。
`null` 表示该轴固定；所有 state_dict key 必须显式列出，标量 buffer 使用 `[]`。
连续特征块可以用 `{"group": "conv_channels", "block_size": 9}` 代替 group 字符串。
BatchNorm 的 weight、bias、running_mean、running_var 应使用对应通道 group，
num_batches_tracked 使用 `[]`。残差两侧需要共享输出 group。
同一 group 出现在同一张量多个轴上时不支持；分组卷积、注意力等特殊结构需要
按其真实对称性设计配置，不能直接按名称猜测。

自定义配置会验证 key、维度、group 大小和排列合法性，但未知网络无法自动前向检查，
需由配置作者验证 B/C 函数等价。若需开箱即用支持更多架构，可扩展
`permute_models.py` 的 `make_model` / `builtin_spec` 并加入前向等价性测试。

测试：

```bash
python3 -m pytest test/test_plot_layer_weight_pca.py test/test_model_comparison_tools.py -q
```
