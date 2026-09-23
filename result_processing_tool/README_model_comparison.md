# 模型权重比较与排列对齐

## 查看或覆盖 checkpoint 的模型/数据集名称

旧 checkpoint 如果保存了已经从新代码移除的名称，可以用
`override_model_pt_info.py` 只修改顶层 metadata，不改变任何权重：

```bash
# 打印当前名称，然后交互式询问是否修改（回答 n 会保持不变）
python3 result_processing_tool/override_model_pt_info.py MODELS

# 原地覆盖模型名和数据集名
python3 result_processing_tool/override_model_pt_info.py MODELS \
  --model-name bnn_floating --dataset-name cifar10 --in-place

# 保留原目录，写入另一个目录；支持递归目录
python3 result_processing_tool/override_model_pt_info.py MODELS \
  --recursive --model-name binary_attention_cct_7_3x1_32 \
  --output-dir MODELS_RETAGGED
```

目标名称使用当前代码中定义的 `ModelType` 和 `DatasetType`，输入旧名称可以不存在。
不指定 `--model-name` / `--dataset-name` 时，脚本会在打印后交互式询问是否修改模型名或数据集名，
再询问原地覆盖还是写到新目录；两个修改问题都回答 `n`，或者取消输出目录时，不会改动文件。
如果连输入目录参数也省略，脚本会先交互式询问目录。
使用 `--model-name` / `--dataset-name` 时仍可通过 `--in-place` 或 `--output-dir` 非交互式执行。
显式输出目录中已有文件会被拒绝覆盖。

在项目根目录运行以下命令。使用生成模型时的 Python 环境，依赖为
`torch`、`numpy`；PCA 另需 `matplotlib`，排列对齐另需 `scipy` 和项目的模型依赖。
只读取 checkpoint，不下载或加载训练数据集。文件格式复用
`py_src.model_opti_save_load`，默认匹配 `*.model.pt`，不读取 optimizer 文件。

## 所有模型对的逐层 cosine similarity

```bash
python3 result_processing_tool/calculate_pairwise_layer_cosine.py \
  tool/high_accuracy_BNN
```

默认在输入目录输出以下三个文件：`layer_cosine_summary.csv`（一层一行的统计）、
`layer_cosine_pairs.csv`（每一层、每一个模型 pair 的明细）和
`layer_cosine_distribution.pdf`（一页 PDF，每个 layer 一个 histogram subplot）。
N 个模型计算 N(N−1)/2 个不重复模型对：20 个模型为 190 对。
`mean_cosine` 为这些 pair-wise cosine 的算术平均值，不是平均权重的 cosine。
同时记录模型数、参数数、总 pair 数、有效/未定义 pair 数、总体标准差、最小值、最大值。
任一模型该层范数为零时，该 pair 的 cosine 为 NaN，排除出平均值并计数；
没有有效 pair 时统计值为 NaN。

也可以用 `-o`、`--pairs-file`、`--distribution-file` 指定输出路径；未指定的输出仍然
使用输入目录中的上述固定文件名。默认输出可以重复运行并覆盖，显式指定的已存在路径
仍会被拒绝，以避免误覆盖其他结果。
默认将同一模块的 weight 和 bias 拼接成一个向量，排除 BatchNorm 的运行统计量。
这与 `plot_layer_weight_pca.py` 的分组规则一致。
当 checkpoint 的 `model_name` 为 `bnn` 时，卷积层和全连接层的 latent weight 会先按
训练时的规则转换为 `-1/+1`（零值也转换为 `-1`），再计算 cosine；BatchNorm 参数仍使用
原始浮点值。`bnn_floating` 和 `binary_attention_cct_7_3x1_32` 的 checkpoint 权重保持浮点计算，
因为它们的权重本身不是二值权重。

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

内置支持 `bnn`、`bnn_floating`、`lenet4`、`lenet5`、`lenet5_large_fc`、
`cct_7_3x1_32`、`binary_attention_cct_7_3x1_32`，自动读取 checkpoint 的
`model_name`。只有元信息缺失时才需 `--model-type`。
构建模型时复用 `ml_setup` 使用的模型类，不构建完整数据集 setup。

算法默认使用 `--method auto`：CNN/MLP 使用
[Git Re-Basin 的 weight matching 思路](https://arxiv.org/abs/2209.04836)，
Binary CCT7 使用下面的 signed matcher。也可以显式传 `--method git_rebasin`
或 `--method signed`；后者目前只支持 `binary_attention_cct_7_3x1_32`。
两种方法都只变换 checkpoint，不重新训练模型：

```bash
python3 result_processing_tool/permute_models.py \
  -a MODELS/00.model.pt -b MODELS -o ALIGNED --method signed
```

Git Re-Basin 方法：
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

Binary CCT7 使用 `py_src/ml_setup_model/bnn/binary_cct.py` 中的
`binary_cct_7_3x1_32` 工厂，支持项目默认的 32×32、10 类、learnable positional embedding 配置：

```bash
python3 result_processing_tool/permute_models.py \
  -a MODELS/00.model.pt -b MODELS -o ALIGNED
```

当 checkpoint 元信息缺失时，可加 `--model-type binary_attention_cct_7_3x1_32`。
signed matcher 先用普通的合法通道排列做基础匹配，然后在每个 attention block 内优化
signed Q/K 和 V/projection 对称性。Q/K 使用耦合的 feature sign，V 的 sign 同步补偿
projection column；Q/K feature permutation 与 V feature permutation 独立，避免把
transformer 的两种 feature 对称性错误地绑在一起。它有 22 个普通排列组：一个贯穿 tokenizer、位置编码、LayerNorm、所有残差分支、
attention pooling 和分类器输入的 embedding 排列，以及每个 transformer block 的
MLP hidden 和 attention heads 排列；signed 阶段另外对每个 head 的 Q/K 与 V feature
分别做 signed assignment。这些变换是 attention 的合法对称性，
不进行会改变函数的任意独立 Q/K/V 行排列。
注意力偏置的 head 轴与 attention heads 同步，token/空间位置及输出类别顺序固定。
匹配时把 packed QKV 和 attention projection 临时视为带 head 轴的张量；
保存时恢复原始 state_dict 的 key、shape 和 dtype，可直接加载到 BinaryCCT7_3x1。
不同类别数、图像尺寸或缺少位置编码的变体不属于此内置配置。

标准 CCT7 使用 `py_src/third_party/compact_transformers/src/cct.py` 中的
`cct_7_3x1_32` 工厂，同样支持项目默认的 32×32、10 类、learnable positional embedding 配置。
它使用 Git Re-Basin 方法；CCT 的 tokenizer、残差 embedding、attention heads/head features、
MLP hidden units 会按合法的网络对称性同步排列。元信息缺失时可加
`--model-type cct_7_3x1_32`。

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
