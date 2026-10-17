# ViT-B/32 architecture comparisons

Two independent models are available through `--model_type` in
`misalignment_measurement_2.py` and `-m` in `generate_high_accuracy_model.py`:

| Model name | Parameters (1000 classes) | Change from `vit_b_32` |
| --- | --- | --- |
| `vit_b_32` | 88,224,232 | Baseline. |
| `vit_b_32_rope` | 88,195,048 | Replace absolute positional embeddings with 2D RoPE-Mixed. |
| `vit_b_32_swiglu` | 88,236,520 | Replace GELU MLPs with SwiGLU, using hidden width 2048 instead of 3072. |

Both are randomly initialized 224px ViT-B/32 models with 12 blocks, 12 attention
heads, embedding width 768, 49 patch tokens, one CLS token and no registers.
They are separate ablations: RoPE keeps GELU; SwiGLU keeps the original attention
and learned absolute positional embeddings. Input size remains fixed at 224px,
as in the baseline; resolution extrapolation is not exposed by these factories.

## RoPE details

The implementation follows the RoPE-Mixed frequency initialization and 2D
coordinates in the [official RoPE-ViT implementation](https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py):
theta is 10, with independently initialized, learnable x/y frequencies per
layer and head. Only patch Q/K vectors are rotated; CLS Q/K and all V vectors
are left unchanged. There is no absolute positional embedding or additional
LayerScale. The backbone otherwise keeps the existing torchvision architecture.
At equal seeds every common parameter starts identically to `vit_b_32`.

Phases and rotations are computed in FP32 under AMP. Attention uses PyTorch
SDPA, and rotations use real arithmetic to support `torch.compile` without
complex-number kernels. There are 9,216 frequency parameters, replacing 38,400
absolute-position parameters (a net reduction of 29,184).

## SwiGLU details

The FFN computes `w3(silu(gate) * value)` with a combined gate/value projection.
The 2/3 hidden-width adjustment follows the
[DINOv2 implementation](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/swiglu_ffn.py).
Widths 3072 (GELU) and 2048 (SwiGLU) have equal FFN matrix parameter counts.
Biases add 12,288 parameters across the 12 blocks. Projection weights use
torchvision's MLP initialization scheme (Xavier uniform, bias normal std 1e-6).
Common non-FFN parameters preserve baseline initialization at equal seeds.
The existing FFN dropout sites are retained; the factory's dropout is zero.
The implementation uses ordinary PyTorch operators and does not require xFormers.

## Running the measurements

Run from the repository root. This matches the earlier 60-epoch comparison's
explicit training settings; change the update-rule arguments as needed.

```bash
for model in vit_b_32_rope vit_b_32_swiglu; do
  python3 tool/misalignment_measurement_2.py \
    --dataset imagenet1k --model_type "$model" \
    --random_seed 1 --split_seed 2026 --directions both \
    --augmentation none --batch_size 128 --epochs 60 \
    --optimizer adamw --learning_rate 5e-5 --weight_decay 0.01 \
    --scheduler fixed --warmup_epochs 5 \
    --train_gradient_weight 0.8 --val_gradient_weight 0.2 \
    --relative_flatness_layers last_matrix_weight \
    --relative_flatness_batch_size 32 \
    --relative_flatness_batches 16 --relative_flatness_samples 8 \
    --token_norm_threshold 150 \
    -o "tool/results/${model}_seed1_60epoch"
done
```

Both models reuse the baseline MLSetup recipe and preset-dependent criterion,
DALI loader, optimizer/scheduler presets, token-norm hooks, checkpoint loading,
and misalignment update/measurement rules. Explicit measurement CLI parameters
continue to override the training preset. ViT measurement defaults to
`--model_preset 3`, preserving the historical label smoothing of 0.11;
`--model_preset 1` explicitly selects zero smoothing. Image augmentation is
controlled separately by `--augmentation`; batch Mixup/CutMix and the
repeated-augmentation sampler are not used. See the
[ImageNet recipe comparison](../../py_src/ml_setup_dataset/README_imagenet_augmentation.md).
The last matrix remains `heads.head.weight` with shape `[1000, 768]`; token
norms are still measured at `encoder.ln.input` for `patch` and `cls`.

Checkpoint metadata stores the respective new model name, allowing strict
reconstruction through `get_ml_setup_from_config`. These are local variants,
not loaders for upstream pretrained RoPE-ViT or DINOv2 checkpoints.
